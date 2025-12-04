import numpy as np
import scipy.io.wavfile as wav


def fft_recursive(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]

    if N <= 1:
        return x
    
    even = fft_recursive(x[::2])
    odd = fft_recursive(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)

    return np.concatenate([
        even + factor[:N//2] * odd,
        even - factor[:N//2] * odd
    ])


def ifft_recursive(X):
    X = np.asarray(X, dtype=complex)
    return np.conjugate(fft_recursive(np.conjugate(X))) / len(X)


def stft(signal, win_size=1024, hop_size=256):
    window = np.hanning(win_size)
    frames = []

    for start in range(0, len(signal) - win_size, hop_size):
        frame = signal[start:start + win_size] * window
        frames.append(fft_recursive(frame))

    return np.array(frames), window


def istft(spectrogram, window, win_size=1024, hop_size=256):
    output_len = hop_size * (len(spectrogram) + 1) + win_size
    result = np.zeros(output_len)
    window_sum = np.zeros(output_len)

    idx = 0
    for frame_fft in spectrogram:
        frame = np.real(ifft_recursive(frame_fft))

        result[idx:idx+win_size] += frame * window
        window_sum[idx:idx+win_size] += window ** 2
        idx += hop_size

  
    nonzero = window_sum > 1e-8
    result[nonzero] /= window_sum[nonzero]

    return result



def noise_reduction_manual_fft(
        input_wav,
        output_wav,
        noise_frames=40,
        alpha=2.5,     
        beta=0.02      
    ):


    rate, data = wav.read(input_wav)


    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0

    win = 1024
    hop = 256

   
    spec, window = stft(data, win_size=win, hop_size=hop)
    mag = np.abs(spec)
    phase = np.angle(spec)

    
    noise_profile = np.mean(mag[:noise_frames, :], axis=0)

    
    eps = 1e-8
    sub = mag - alpha * noise_profile[np.newaxis, :]

    
    gain = np.maximum(sub / (mag + eps), beta)

    
    spec_clean = gain * mag * np.exp(1j * phase)

     
    cleaned = istft(spec_clean, window, win_size=win, hop_size=hop)

    
    cleaned = np.clip(cleaned, -1, 1)
    cleaned = (cleaned * 32767).astype(np.int16)

    wav.write(output_wav, rate, cleaned)
    print("Готово! Файл сохранён:", output_wav)



if __name__ == "__main__":
    noise_reduction_manual_fft(
        "input_test_2.wav",
        "output_result_2.wav",
        noise_frames=100,
        alpha=3.5,
        beta=0.005
    )
