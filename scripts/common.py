"""Ortak yardimcilar: cozunurluk hesabi, yollar, latent boyutlari."""

from dataclasses import dataclass

# SD1.5 sabitleri
LATENT_CHANNELS = 4
VAE_SCALE_FACTOR = 8          # 512 px -> 64 latent
TEXT_SEQ_LEN = 77
TEXT_HIDDEN = 768             # CLIP ViT-L/14


@dataclass(frozen=True)
class Resolution:
    """Bir uretim cozunurlugu. NPU sabit (static) sekil ister, bu yuzden her
    cozunurluk icin ayri bir UNet grafi derlenir."""
    width: int
    height: int

    @property
    def latent_w(self) -> int:
        return self.width // VAE_SCALE_FACTOR

    @property
    def latent_h(self) -> int:
        return self.height // VAE_SCALE_FACTOR

    @property
    def tag(self) -> str:
        return f"{self.width}x{self.height}"


# Local Dream'in SD1.5 icin paketledigi standart uc cozunurluk
DEFAULT_RESOLUTIONS = [
    Resolution(512, 512),
    Resolution(512, 768),
    Resolution(768, 512),
]


def parse_resolutions(spec: str):
    """'512x512,512x768' -> [Resolution, ...]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        w, h = part.lower().split("x")
        out.append(Resolution(int(w), int(h)))
    return out
