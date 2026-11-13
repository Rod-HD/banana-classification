"""Huấn luyện CNN (hướng A và hướng B). Adam + cross-entropy, dừng sớm theo macro-F1 trên val,
lưu checkpoint mỗi epoch vào checkpoints/<run>/ và hỗ trợ --resume. Khi xong, mỗi run để lại
checkpoints/<run>/best.keras (kiến trúc + trọng số tốt nhất) và state.json; run đã xong thì --resume không huấn luyện lại.

Ví dụ:
    python -m src.cnn.train --model a_lenet
    python -m src.cnn.train --model a_dropout --no-dropout --tag nodrop
    python -m src.cnn.train --model a_vgg16 --frac 0.25
    python -m src.cnn.train --model b_mobilenetv3 --stage 1
    python -m src.cnn.train --model b_mobilenetv3 --stage 2 --lr 1e-5 --augment
Kết quả: results/cnn/<run>/ (metrics.json, classification_report.txt, confusion_matrix.png, history.json, curves.png).
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import cv2
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from ..config import CACHE_DIR, CKPT_DIR, RESULTS_DIR, SEED
from ..data import load_all_images, load_split
from ..metrics import save_result
from . import models as M

DEFAULTS = {  # lr, số epoch tối đa, patience
    "a_lenet": (1e-3, 60, 10), "a_dropout": (1e-3, 60, 10), "a_vgg16": (1e-4, 40, 8),
    "b_s1": (1e-3, 60, 10), "b_s2": (1e-5, 20, 5),
}


def setup(seed: int = SEED):
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def images_at(size: int) -> np.ndarray:
    """Toàn bộ ảnh resize về size×size (từ bản 256×256 đã cache), cache ra data/cache/."""
    if size == 256:
        return load_all_images(256)
    f = CACHE_DIR / f"images_{size}.npy"
    if not f.exists():
        src = load_all_images(256)
        np.save(f, np.stack([cv2.resize(np.asarray(im), (size, size), interpolation=cv2.INTER_AREA) for im in src]))
    return np.load(f, mmap_mode="r")


def resize_batch(imgs: np.ndarray, size: int) -> np.ndarray:
    if imgs.shape[1] == size:
        return np.asarray(imgs)
    return np.stack([cv2.resize(np.asarray(im), (size, size), interpolation=cv2.INTER_AREA) for im in imgs])


def backbone_features(kind: str, batch: int = 32) -> np.ndarray:
    """Đầu ra của thân mạng đóng băng cho toàn bộ ảnh (tính một lần, cache).
    a_vgg16: bản đồ 7×7×512; hướng B: vector sau GlobalAveragePooling."""
    f = CACHE_DIR / f"cnnfeat_{kind}.npy"
    if f.exists():
        return np.load(f, mmap_mode="r")
    net = backbone_net(kind)
    X = images_at(224)
    t0 = time.perf_counter()
    out = np.concatenate([net.predict(np.asarray(X[a:a + 256]).astype(np.float32), batch_size=batch, verbose=0)
                          for a in range(0, len(X), 256)])
    np.save(f, out)
    f.with_suffix(".time").write_text(str(time.perf_counter() - t0))
    return np.load(f, mmap_mode="r")


def backbone_net(kind: str):
    if kind == "a_vgg16":
        return M.vgg16_features()
    full, base = M.b_model(kind, augment=False)
    return keras.Model(full.input, full.get_layer("embedding").output)


def train_indices(frac: float, oversample: bool = False):
    tr = load_split("train")
    if frac < 1.0:
        tr, _ = train_test_split(tr, train_size=frac, stratify=tr["label_id"], random_state=SEED)
        tr = tr.sort_values("idx")
    idx, y = tr["idx"].values, tr["label_id"].values
    if oversample:   # lặp lại ngẫu nhiên ảnh của lớp ít cho tới khi bằng lớp đông nhất (chỉ trên train)
        rng = np.random.default_rng(SEED)
        n = np.bincount(y).max()
        sel = np.concatenate([np.r_[np.where(y == c)[0], rng.choice(np.where(y == c)[0], n - (y == c).sum(), replace=True)]
                              for c in range(M.N_CLASSES)])
        idx, y = idx[sel], y[sel]
    return idx, y


def minority_augmented(idx, y, size: int):
    """Tăng cường ảnh riêng cho các lớp ít ảnh (I2): với mỗi lớp chưa bằng lớp đông nhất, tạo thêm
    bản lật/xoay ngẫu nhiên (±30°) của ảnh train thuộc lớp đó cho tới khi đủ số lượng."""
    from ..transforms import rotate
    rng = np.random.default_rng(SEED)
    n = np.bincount(y).max()
    X = images_at(size)
    imgs, labels = [], []
    for c in range(M.N_CLASSES):
        pool = idx[y == c]
        for i in rng.choice(pool, n - len(pool), replace=True):
            im = np.asarray(X[i])
            if rng.random() < 0.5: im = im[:, ::-1]
            if rng.random() < 0.5: im = im[::-1]
            imgs.append(rotate(np.ascontiguousarray(im), angle=float(rng.uniform(-30, 30)))); labels.append(c)
    return np.stack(imgs), np.array(labels)


class F1Checkpoint(keras.callbacks.Callback):
    """Sau mỗi epoch: tính loss, accuracy, macro-F1 trên val; lưu trọng số tốt nhất theo macro-F1;
    lưu checkpoint đầy đủ để --resume; dừng sớm khi macro-F1 không tăng sau `patience` epoch."""

    def __init__(self, Xva, yva, ckpt, state, patience, train_eval=None):
        super().__init__()
        self.Xva, self.yva, self.ckpt, self.state, self.patience, self.train_eval = Xva, yva, ckpt, state, patience, train_eval
        self.t0 = None

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        st = self.state
        st["train_time_s"] += time.perf_counter() - self.t0
        p = self.model.predict(self.Xva, batch_size=64, verbose=0)
        pred = p.argmax(1)
        rec = {"epoch": epoch + 1, "loss": float(logs["loss"]), "accuracy": float(logs["accuracy"]),
               "val_loss": float(-np.log(np.clip(p[np.arange(len(pred)), self.yva], 1e-7, 1)).mean()),
               "val_accuracy": float(accuracy_score(self.yva, pred)),
               "val_macro_f1": float(f1_score(self.yva, pred, average="macro"))}
        if self.train_eval is not None:   # loss/accuracy trên train ở chế độ suy luận (tắt dropout, tăng cường)
            Xt, yt = self.train_eval
            pt = self.model.predict(Xt, batch_size=64, verbose=0)
            rec["train_eval_loss"] = float(-np.log(np.clip(pt[np.arange(len(yt)), yt], 1e-7, 1)).mean())
            rec["train_eval_accuracy"] = float(accuracy_score(yt, pt.argmax(1)))
        st["history"].append(rec); st["epoch"] = epoch + 1
        if rec["val_macro_f1"] > st["best_f1"] + 1e-6:
            st["best_f1"], st["best_epoch"], st["wait"] = rec["val_macro_f1"], epoch + 1, 0
            self.model.save_weights(self.ckpt / "best.weights.h5")
        else:
            st["wait"] += 1
        if st["wait"] >= self.patience:
            st["done"] = True
            self.model.stop_training = True
        self.model.save(self.ckpt / "last.keras")
        (self.ckpt / "state.json").write_text(json.dumps(st))
        print(f"epoch {epoch + 1:3d} loss {rec['loss']:.4f} acc {rec['accuracy']:.4f} | val loss {rec['val_loss']:.4f} "
              f"acc {rec['val_accuracy']:.4f} macro-F1 {rec['val_macro_f1']:.4f} | best {st['best_f1']:.4f} @ {st['best_epoch']}", flush=True)


def save_inference(model, path):
    """Lưu mô hình ở dạng chỉ gồm kiến trúc và trọng số, không kèm trạng thái của bộ tối ưu Adam
    (hai mảng moment cho mỗi trọng số huấn luyện được, chỉ cần khi huấn luyện tiếp và làm file lớn gấp ba).
    Đây là file dùng để dự đoán và là file được đo dung lượng."""
    tmp = path.with_name("slim.tmp.keras")
    model.save(tmp)
    keras.models.load_model(tmp, compile=False).save(path)
    tmp.unlink()


def fit_loop(model, Xtr, ytr, Xva, yva, run, lr, epochs, patience, batch=32, resume=False, class_weight=None,
             train_eval=None):
    ckpt = CKPT_DIR / run; ckpt.mkdir(parents=True, exist_ok=True)
    state = {"epoch": 0, "best_f1": -1.0, "best_epoch": 0, "wait": 0, "history": [], "train_time_s": 0.0, "done": False}
    best_w, best_k = ckpt / "best.weights.h5", ckpt / "best.keras"
    saved = json.loads((ckpt / "state.json").read_text()) if resume and (ckpt / "state.json").exists() else None
    if saved and saved.get("done") and (best_k.exists() or best_w.exists()):
        state = saved          # run đã xong: không huấn luyện lại, chỉ nạp trọng số tốt nhất
        print(f"run đã hoàn tất ({state['epoch']} epoch, best {state['best_f1']:.4f}); dùng lại trọng số đã lưu", flush=True)
    elif saved and (ckpt / "last.keras").exists():
        state = saved
        model = keras.models.load_model(ckpt / "last.keras")
        print(f"tiếp tục từ epoch {state['epoch']} (best {state['best_f1']:.4f})", flush=True)
    else:
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    if not state["done"] and state["epoch"] < epochs:
        ds = (tf.data.Dataset.from_tensor_slices((Xtr, ytr)).shuffle(len(Xtr), seed=SEED, reshuffle_each_iteration=True)
              .batch(batch).map(lambda x, y: (tf.cast(x, tf.float32), y)).prefetch(tf.data.AUTOTUNE))
        cb = F1Checkpoint(Xva, yva, ckpt, state, patience, train_eval)
        model.fit(ds, epochs=epochs, initial_epoch=state["epoch"], callbacks=[cb], verbose=0, class_weight=class_weight)
    state["done"] = True
    (ckpt / "state.json").write_text(json.dumps(state))
    if best_w.exists():                     # vừa huấn luyện xong: chuyển trọng số tốt nhất thành best.keras
        model.load_weights(best_w)
        save_inference(model, best_k)
        best_w.unlink()
    else:                                   # run đã xong từ trước (best.keras có sẵn trong repo)
        model.load_weights(best_k)
    if (ckpt / "last.keras").exists():      # checkpoint để --resume chỉ cần khi run còn dang dở
        (ckpt / "last.keras").unlink()
    return model, state


def plot_curves(history, path, title):
    ep = [h["epoch"] for h in history]
    best = max(history, key=lambda h: h["val_macro_f1"])["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    tl = "train_eval_loss" if "train_eval_loss" in history[0] else "loss"
    ta = "train_eval_accuracy" if "train_eval_accuracy" in history[0] else "accuracy"
    axes[0].plot(ep, [h[tl] for h in history], label="train"); axes[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss"); axes[0].set_yscale("log")
    axes[1].plot(ep, [h[ta] for h in history], label="train"); axes[1].plot(ep, [h["val_accuracy"] for h in history], label="val")
    axes[1].set_title("Accuracy")
    axes[2].plot(ep, [h["val_macro_f1"] for h in history], color="tab:green", label="val macro-F1"); axes[2].set_title("Macro-F1 trên val")
    for ax in axes:
        ax.axvline(best, color="gray", ls="--", lw=1); ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
    fig.suptitle(f"{title} (đường đứt: epoch được chọn = {best})")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def run_name(a) -> str:
    n = a.model
    if a.model.startswith("b_") and a.stage == 1:
        n += "_s1"
    if a.frac < 1.0:
        n += f"_frac{int(round(a.frac * 100))}"
    if a.tag:
        n += f"_{a.tag}"
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(M.INPUT_SIZE))
    ap.add_argument("--stage", type=int, default=2, choices=[1, 2], help="chỉ dùng cho hướng B: 1 = đóng băng thân, 2 = fine-tune")
    ap.add_argument("--frac", type=float, default=1.0, help="tỉ lệ tập train được dùng (N2)")
    ap.add_argument("--size", type=int, default=None, help="kích thước ảnh đầu vào (mặc định theo mô hình)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--augment", action="store_true", help="bật tăng cường ảnh lúc huấn luyện")
    ap.add_argument("--no-dropout", action="store_true", help="a_dropout: bỏ lớp Dropout (N3)")
    ap.add_argument("--class-weight", action="store_true", help="trọng số lớp nghịch với tần suất (I2)")
    ap.add_argument("--oversample", action="store_true", help="lặp lại ảnh lớp ít trên tập train (I2)")
    ap.add_argument("--minority-aug", action="store_true", help="thêm bản tăng cường cho các lớp ít ảnh trên tập train (I2)")
    ap.add_argument("--train-eval", action="store_true", help="mỗi epoch chấm thêm tập train ở chế độ suy luận (N3, N5)")
    ap.add_argument("--init-from", default=None, help="hướng B giai đoạn 2: tên run giai đoạn 1 để lấy đầu phân lớp")
    ap.add_argument("--seed", type=int, default=SEED, help="seed khởi tạo trọng số và thứ tự lô (cách chia dữ liệu không đổi)")
    ap.add_argument("--tag", default="", help="hậu tố tên run")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    setup(a.seed)
    run = run_name(a)
    size = a.size or M.INPUT_SIZE[a.model]
    key = a.model if a.model.startswith("a_") else f"b_s{a.stage}"
    lr, epochs, patience = (a.lr or DEFAULTS[key][0], a.epochs or DEFAULTS[key][1], a.patience or DEFAULTS[key][2])
    idx_tr, ytr = train_indices(a.frac, a.oversample)
    va = load_split("val"); idx_va, yva = va["idx"].values, va["label_id"].values
    cw = None
    if a.class_weight:
        cnt = np.bincount(ytr, minlength=M.N_CLASSES)
        cw = {c: float(len(ytr) / (M.N_CLASSES * cnt[c])) for c in range(M.N_CLASSES)}
    frozen = a.model == "a_vgg16" or (a.model.startswith("b_") and a.stage == 1)
    t_feat = 0.0
    if frozen:      # thân đóng băng: dùng đặc trưng tính sẵn, chỉ huấn luyện phần đầu
        F = backbone_features(a.model)
        t_feat = float((CACHE_DIR / f"cnnfeat_{a.model}.time").read_text())
        Xtr, Xva = np.asarray(F[idx_tr]), np.asarray(F[idx_va])
        model = M.a_vgg16_head(F.shape[1:]) if a.model == "a_vgg16" else M.b_head(F.shape[1])
        base = None
    else:
        X = images_at(size)
        Xtr, Xva = np.asarray(X[idx_tr]), np.asarray(X[idx_va]).astype(np.float32)
        if a.model == "a_lenet":
            model = M.a_lenet(size, augment=a.augment)
        elif a.model == "a_dropout":
            model = M.a_dropout(size, dropout=not a.no_dropout, augment=a.augment)
        else:       # hướng B, giai đoạn 2
            model, base = M.b_model(a.model, size, augment=a.augment)
            init = a.init_from or (a.model + "_s1" + (f"_frac{int(round(a.frac * 100))}" if a.frac < 1.0 else ""))
            head = M.b_head(base.output_shape[-1]); head.load_weights(CKPT_DIR / init / "best.keras")
            model.get_layer("softmax").set_weights(head.get_layer("softmax").get_weights())
            M.unfreeze(base, a.model)
    if a.minority_aug:
        Xa, ya = minority_augmented(idx_tr, ytr, 224 if frozen else size)
        if frozen:
            Xa = backbone_net(a.model).predict(Xa.astype(np.float32), batch_size=32, verbose=0)
        Xtr, ytr = np.concatenate([Xtr, Xa]), np.concatenate([ytr, ya])
    train_eval = (Xtr.astype(np.float32) if not frozen else Xtr, ytr) if a.train_eval else None
    print(f"run={run} n_train={len(ytr)} n_val={len(yva)} size={size} lr={lr} epochs={epochs} patience={patience} "
          f"params={M.count_params(model)}", flush=True)
    model, st = fit_loop(model, Xtr, ytr, Xva, yva, run, lr, epochs, patience, a.batch, a.resume, cw, train_eval)
    t0 = time.perf_counter(); pva = model.predict(Xva, batch_size=64, verbose=0); t_pred = time.perf_counter() - t0
    ptr = model.predict(Xtr.astype(np.float32) if not frozen else Xtr, batch_size=64, verbose=0).argmax(1)
    hist = st["history"]; best = hist[st["best_epoch"] - 1]
    extra = {"run": run, "model": a.model, "stage": a.stage if a.model.startswith("b_") else None, "eval_set": "val",
             "input_size": size, "frac": a.frac, "n_train": int(len(ytr)), "lr": lr, "batch": a.batch,
             "augment": bool(a.augment), "dropout": not a.no_dropout, "class_weight": bool(a.class_weight), "oversample": bool(a.oversample), "minority_aug": bool(a.minority_aug),
             "seed": a.seed, "frozen_backbone": bool(frozen), "params": M.count_params(model),
             "epochs_run": st["epoch"], "best_epoch": st["best_epoch"], "train_time_s": st["train_time_s"],
             "time_per_epoch_s": st["train_time_s"] / max(st["epoch"], 1), "backbone_feature_time_s_all_2471_images": t_feat,
             "val_loss": best["val_loss"], "train_accuracy_inference_mode": float(accuracy_score(ytr, ptr)),
             "train_macro_f1_inference_mode": float(f1_score(ytr, ptr, average="macro")),
             "predict_ms_per_image_batched_val": 1000 * t_pred / len(yva),
             "best_model_file_mb": os.path.getsize(CKPT_DIR / run / "best.keras") / 1e6}
    out = RESULTS_DIR / "cnn" / run
    m = save_result(out, yva, pva.argmax(1), extra=extra, title=f"{run} (val)")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    plot_curves(hist, out / "curves.png", run)
    print(f"KẾT QUẢ {run}: val macro-F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f} | train acc {extra['train_accuracy_inference_mode']:.4f} "
          f"| {st['epoch']} epoch, {st['train_time_s']:.0f}s ({extra['time_per_epoch_s']:.1f}s/epoch)", flush=True)
    return m


if __name__ == "__main__":
    main()
