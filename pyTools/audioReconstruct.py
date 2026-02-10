import numpy as np
from PIL import Image
from scipy.io import wavfile

# ----------------------------
# Utilities: auto-crop plot area
# ----------------------------
def auto_crop_plot_area(gray, bg_quantile=0.90, pad=2):
    """
    Try to crop to the plotted spectrogram rectangle by removing margins/labels.
    Assumes background is light and plotted region contains darker ink.
    """
    g = gray.astype(np.float32)

    # Estimate "background brightness" and find pixels significantly darker than it
    bg = np.quantile(g, bg_quantile)
    mask = g < (bg - 5)  # 5 is small; increase if lots of noise in margins

    # If mask is too small, just return original
    if mask.sum() < 100:
        return gray

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # Pad and clip
    y0 = max(0, y0 - pad)
    y1 = min(gray.shape[0], y1 + pad + 1)
    x0 = max(0, x0 - pad)
    x1 = min(gray.shape[1], x1 + pad + 1)

    return gray[y0:y1, x0:x1]


# ----------------------------
# Convert image -> magnitude spectrogram
# ----------------------------
def image_to_mag(
    image_path,
    f_min=0.0,
    f_max=10_000.0,
    duration_s=3.0,
    sr=22050,
    n_fft=2048,
    db_range=70.0,
    gamma=1.0,
    crop="auto",
    crop_box=None,
):
    """
    Returns magnitude spectrogram (freq_bins x frames) suitable for Griffin-Lim.

    Parameters:
      - f_min/f_max: y-axis frequency mapping (Hz)
      - duration_s: total x-axis time span (seconds)
      - sr: target sampling rate
      - n_fft: FFT size (freq bins = n_fft//2 + 1)
      - db_range: darkest pixel corresponds to 0 dB, lightest to -db_range dB
      - gamma: contrast shaping (>1 emphasizes dark traces; <1 boosts faint content)
      - crop: "auto", "none", or "box"
      - crop_box: (left, upper, right, lower) in PIL coordinates if crop="box"
    """
    img = Image.open(image_path).convert("L")
    if crop == "box" and crop_box is not None:
        img = img.crop(crop_box)

    gray = np.array(img)

    if crop == "auto":
        gray = auto_crop_plot_area(gray)

    # Normalize brightness: 0..1 where 1 = dark (loud), 0 = light (quiet)
    g = gray.astype(np.float32) / 255.0
    loud = 1.0 - g  # invert: dark ink -> higher
    loud = np.clip(loud, 0.0, 1.0)

    # Optional contrast shaping
    if gamma != 1.0:
        loud = loud ** gamma

    # Image is (H x W). Top row is f_max, bottom is f_min for typical spectrogram plots.
    H, W = loud.shape

    # Map image rows to target STFT frequency bins (linear in Hz).
    n_freq = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freq)

    # Which image row corresponds to each freq?
    # y=0 at top -> f_max, y=H-1 at bottom -> f_min
    # freq = f_max - (y/(H-1))*(f_max-f_min)
    # => y = (f_max - freq)/(f_max-f_min) * (H-1)
    eps = 1e-9
    y_for_freq = (f_max - freqs) / (max(eps, (f_max - f_min))) * (H - 1)
    y_for_freq = np.clip(y_for_freq, 0, H - 1)

    # Sample the image vertically for each freq bin (nearest-neighbor)
    y_idx = np.rint(y_for_freq).astype(int)
    spec_from_img = loud[y_idx, :]  # (n_freq x W)

    # Map columns to frames: choose hop_length so that W frames spans duration_s
    # frames ~ W, samples ~ duration_s*sr, hop ~ samples/(W-1)
    total_samples = int(round(duration_s * sr))
    hop_length = max(1, int(round(total_samples / max(1, (W - 1)))))

    # We will use exactly W frames.
    mag_linear = spec_from_img

    # Convert 0..1 to dB in [-db_range, 0], then to linear magnitude
    # 1 -> 0 dB, 0 -> -db_range dB
    db = (-db_range) * (1.0 - mag_linear)
    mag = 10.0 ** (db / 20.0)

    return mag.astype(np.float32), hop_length


# ----------------------------
# Griffin–Lim (phase reconstruction)
# ----------------------------
def stft(y, n_fft, hop_length, window):
    y = np.asarray(y, dtype=np.float32)
    win = window(n_fft).astype(np.float32)

    if len(y) < n_fft:
        n_frames = 1
    else:
        n_frames = int(np.ceil((len(y) - n_fft) / hop_length)) + 1

    frames = np.zeros((n_fft, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        frames[:, i] = frame * win

    X = np.fft.rfft(frames, axis=0)  # (n_fft//2+1, frames)
    return X


def _match_frames(X, target_frames):
    """Crop or pad spectrogram along time axis to exactly target_frames."""
    cur = X.shape[1]
    if cur == target_frames:
        return X
    if cur > target_frames:
        return X[:, :target_frames]
    # pad by repeating last column
    pad = target_frames - cur
    last = X[:, -1:]
    return np.concatenate([X, np.repeat(last, pad, axis=1)], axis=1)


def griffin_lim(mag, n_fft, hop_length, n_iter=64, length=None, window_fn=np.hanning, seed=0):
    """
    mag: (n_fft//2+1, frames)
    Ensures internal STFT estimates always match mag's frame count.
    """
    target_frames = mag.shape[1]

    # If length not given, set a length consistent with target_frames
    if length is None:
        length = n_fft + hop_length * (target_frames - 1)

    rng = np.random.default_rng(seed)
    angles = np.exp(2j * np.pi * rng.random(mag.shape)).astype(np.complex64)
    X = mag * angles

    for _ in range(n_iter):
        y = istft(X, n_fft=n_fft, hop_length=hop_length, window=window_fn, length=length)
        X_est = stft(y, n_fft=n_fft, hop_length=hop_length, window=window_fn)
        X_est = _match_frames(X_est, target_frames)
        X = mag * np.exp(1j * np.angle(X_est)).astype(np.complex64)

    y = istft(X, n_fft=n_fft, hop_length=hop_length, window=window_fn, length=length)
    return y


def istft(X, n_fft, hop_length, window, length=None):
    win = window(n_fft).astype(np.float32)
    frames = np.fft.irfft(X, n=n_fft, axis=0).astype(np.float32)  # (n_fft, frames)
    frames *= win[:, None]

    n_frames = frames.shape[1]
    y_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(y_len, dtype=np.float32)
    wsum = np.zeros(y_len, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        y[start:start + n_fft] += frames[:, i]
        wsum[start:start + n_fft] += win ** 2

    # Normalize by window overlap-add weight
    nonzero = wsum > 1e-8
    y[nonzero] /= wsum[nonzero]

    if length is not None:
        if len(y) < length:
            y = np.pad(y, (0, length - len(y)))
        else:
            y = y[:length]
    return y




# ----------------------------
# End-to-end helper
# ----------------------------
def reconstruct_audio_from_spectrogram_image(
    image_path,
    out_wav_path,
    f_min=0.0,
    f_max=10_000.0,
    duration_s=3.0,
    sr=22050,
    n_fft=2048,
    db_range=70.0,
    gamma=1.2,
    gl_iters=80,
    crop="auto",
    crop_box=None,
):
    mag, hop = image_to_mag(
        image_path=image_path,
        f_min=f_min,
        f_max=f_max,
        duration_s=duration_s,
        sr=sr,
        n_fft=n_fft,
        db_range=db_range,
        gamma=gamma,
        crop=crop,
        crop_box=crop_box,
    )

    length = int(round(duration_s * sr))
    y = griffin_lim(mag, n_fft=n_fft, hop_length=hop, n_iter=gl_iters, length=length)

    # Normalize to int16 WAV
    y = y / (np.max(np.abs(y)) + 1e-9)
    wavfile.write(out_wav_path, sr, (y * 32767.0).astype(np.int16))
    return out_wav_path


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    import sys
    import os
    # Usage: python audioReconstruct.py input_image.jpg
    input_image = sys.argv[1] if len(sys.argv) > 1 else "bird_spec.jpg"
    print(f"Reconstructing audio from {input_image}...")
    base, _ = os.path.splitext(input_image)
    out_wav = f"{base}_fake.wav"
    # Example for a 0–10 kHz, 3 s plot:
    reconstruct_audio_from_spectrogram_image(
        image_path=input_image,
        out_wav_path=out_wav,
        f_min=0.0,
        f_max=10_000.0,
        duration_s=3.0,
        sr=22050,
        n_fft=2048,
        db_range=70.0,
        gamma=1.3,
        gl_iters=100,
        crop="auto",
    )
    print(f"Wrote {out_wav}")

