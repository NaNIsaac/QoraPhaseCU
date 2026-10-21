import numpy as np
import pytest

N, M = 64, 24


@pytest.fixture(scope="module")
def triangle():
    """The object of the QoraPhase README figure."""
    y, x = np.indices((M, M)) - (M - 1) / 2
    inside = np.indices((M, M)).sum(0) < M
    block = (1 - 0.5 * np.exp(-(x**2 + y**2) / 20)) * np.exp(
        1j * 1.2 * np.exp(-((x + 4) ** 2 + (y + 4) ** 2) / 30)
    )
    obj = np.zeros((N, N), np.complex64)
    obj[N // 2 - M // 2 : N // 2 + M // 2, N // 2 - M // 2 : N // 2 + M // 2] = block * inside
    intensity = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj), norm="ortho"))) ** 2
    return obj, np.abs(obj) > 0, intensity.astype(np.float32)
