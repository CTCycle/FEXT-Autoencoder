from __future__ import annotations

import os
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from tqdm import tqdm

from FEXT.app.client.workers import check_thread_status, update_progress_callback
from FEXT.app.utils.constants import EVALUATION_PATH
from FEXT.app.utils.logger import logger
from FEXT.app.utils.repository.serializer import DataSerializer


# [VALIDATION OF PRETRAINED MODELS]
###############################################################################
class ImageAnalysis:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.serializer = DataSerializer()
        self.img_resolution = 400
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def save_image(self, fig: Figure, name: str) -> None:
        out_path = os.path.join(EVALUATION_PATH, name)
        fig.savefig(out_path, bbox_inches="tight", dpi=self.img_resolution)

    # -------------------------------------------------------------------------
    def load_color_and_gray(
        self, path: str
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Load BGR and grayscale; returns (bgr, gray) or (None, None) if failed."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray

    # -------------------------------------------------------------------------
    def calculate_image_statistics(
        self, images_path: list[str], **kwargs
    ) -> pd.DataFrame:
        results = []
        for i, path in enumerate(
            tqdm(
                images_path, desc="Processing images", total=len(images_path), ncols=100
            )
        ):
            img = cv2.imread(path)
            if img is None:
                logger.warning(f"Warning: Unable to load image at {path}.")
                continue

            # Convert image to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Get image dimensions
            height, width = gray.shape
            # Compute basic statistics
            mean_val = np.mean(gray)
            median_val = np.median(gray)
            std_val = np.std(gray)
            min_val = np.min(gray)
            max_val = np.max(gray)
            pixel_range = max_val - min_val
            # Estimate noise by comparing the image to a blurred version
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            noise = gray.astype(np.float32) - blurred.astype(np.float32)
            noise_std = np.std(noise)
            # Define the noise ratio (avoiding division by zero with a small epsilon)
            noise_ratio = noise_std / (std_val + 1e-9)
            results.append(
                {
                    "name": os.path.basename(path),
                    "height": height,
                    "width": width,
                    "mean": mean_val,
                    "median": median_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                    "pixel_range": pixel_range,
                    "noise_std": noise_std,
                    "noise_ratio": noise_ratio,
                }
            )

            # check for thread status and progress bar update
            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(images_path), kwargs.get("progress_callback", None)
            )

        # create dataframe from calculated statistics and save table into database
        stats_dataframe = pd.DataFrame(results)
        self.serializer.save_images_statistics(stats_dataframe)
        logger.info(f"Image statistics saved: {len(stats_dataframe)} records")

        return stats_dataframe

    # -------------------------------------------------------------------------
    def calculate_pixel_intensity_distribution(
        self, images_path: list[str], **kwargs
    ) -> Figure:
        image_histograms = np.zeros(256, dtype=np.int64)
        for i, path in enumerate(
            tqdm(
                images_path,
                desc="Processing image histograms",
                total=len(images_path),
                ncols=100,
            )
        ):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Warning: Unable to load image at {path}.")
                continue

            # Calculate histogram for grayscale values [0, 255]
            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            image_histograms += hist.astype(np.int64)

            # check for thread status and progress bar update
            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(images_path), kwargs.get("progress_callback", None)
            )

        # Plot the combined pixel intensity histogram
        fig, ax = plt.subplots(figsize=(18, 16), dpi=self.img_resolution)
        plt.bar(np.arange(256), image_histograms, alpha=0.7)
        ax.set_title("Pixel Intensity Histogram", fontsize=24)
        ax.set_xlabel("Pixel Intensity", fontsize=16, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=16, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=14, labelcolor="black")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        plt.tight_layout()
        self.save_image(fig, "pixels_intensity_histogram.jpeg")
        plt.close()

        return fig

    # -------------------------------------------------------------------------
    def calculate_exposure_metrics(
        self,
        images_path: list[str],
        low_thresh: int = 10,
        high_thresh: int = 245,
        **kwargs,
    ) -> pd.DataFrame:
        results: list[dict] = []
        for i, path in enumerate(
            tqdm(
                images_path, desc="Exposure metrics", total=len(images_path), ncols=100
            )
        ):
            _, gray = self.load_color_and_gray(path)
            if gray is None:
                logger.warning(f"Warning: Unable to load image at {path}.")
                continue

            total = gray.size
            under = np.count_nonzero(gray < low_thresh)
            over = np.count_nonzero(gray > high_thresh)
            mid = total - under - over
            mean_gray = float(np.mean(gray))

            results.append(
                {
                    "name": os.path.basename(path),
                    "underexposed_pct": (under / total) * 100.0,
                    "overexposed_pct": (over / total) * 100.0,
                    "midtone_pct": (mid / total) * 100.0,
                    "mean_gray": mean_gray,
                }
            )

            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(images_path), kwargs.get("progress_callback", None)
            )

        data = pd.DataFrame(results)
        self.serializer.save_images_exposure_metrics(data)
        logger.info(f"Exposure metrics computed: {len(data)} records")

        return data

    # -------------------------------------------------------------------------
    def calculate_entropy(
        self, images_path: list[str], bins: int = 256, **kwargs
    ) -> pd.DataFrame:
        results: list[dict] = []
        log2_bins = np.log2(bins)

        for i, path in enumerate(
            tqdm(images_path, desc="Entropy metrics", total=len(images_path), ncols=100)
        ):
            _, gray = self.load_color_and_gray(path)
            if gray is None:
                logger.warning(f"Warning: Unable to load image at {path}.")
                continue

            hist, _ = np.histogram(gray, bins=bins, range=(0, 255))
            p = hist.astype(np.float64) / (gray.size + 1e-12)
            # Shannon entropy (avoid log2(0))
            entropy_bits = -np.sum(np.where(p > 0, p * np.log2(p), 0.0))
            results.append(
                {
                    "name": os.path.basename(path),
                    "entropy_bits": float(entropy_bits),
                    "normalized_entropy": float(entropy_bits / (log2_bins + 1e-12)),
                }
            )

            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(images_path), kwargs.get("progress_callback", None)
            )

        data = pd.DataFrame(results)
        self.serializer.save_images_entropy_metrics(data)
        logger.info(f"Entropy metrics computed: {len(data)} records")

        return data

    # -------------------------------------------------------------------------
    def calculate_sharpness_metrics(
        self, images_path: list[str], **kwargs
    ) -> pd.DataFrame:
        results: list[dict] = []
        for i, path in enumerate(
            tqdm(
                images_path, desc="Sharpness metrics", total=len(images_path), ncols=100
            )
        ):
            _, gray = self.load_color_and_gray(path)
            if gray is None:
                logger.warning(f"Warning: Unable to load image at {path}.")
                continue

            # Laplacian variance
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            var_lap = float(lap.var())

            # Tenengrad (Sobel gradient magnitude squared)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag2 = gx * gx + gy * gy
            tenengrad = float(np.mean(mag2))

            # Spectral high-frequency ratio
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            mag = np.abs(fshift)
            h, w = gray.shape
            yy, xx = np.ogrid[:h, :w]
            cy, cx = h // 2, w // 2
            r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            r_max = np.sqrt((max(cy, h - cy)) ** 2 + (max(cx, w - cx)) ** 2)
            mask_high = r >= (0.25 * r_max)  # keep outer 75% radius as "high"
            total_energy = np.sum(mag) + 1e-12
            high_energy = np.sum(mag[mask_high])
            spectral_highfreq_ratio = float(high_energy / total_energy)

            results.append(
                {
                    "name": os.path.basename(path),
                    "var_laplacian": var_lap,
                    "tenengrad": tenengrad,
                    "spectral_highfreq_ratio": spectral_highfreq_ratio,
                }
            )

            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(images_path), kwargs.get("progress_callback", None)
            )

        data = pd.DataFrame(results)
        self.serializer.save_images_sharpness_metrics(data)
        logger.info(f"Sharpness metrics computed: {len(data)} records")
        return data

    # -------------------------------------------------------------------------
    def calculate_color_metrics(self, images_path: list[str], **kwargs) -> pd.DataFrame:
        results: list[dict] = []
        for i, path in enumerate(
            tqdm(images_path, desc="Color metrics", total=len(images_path), ncols=100)
        ):
            bgr, _ = self.load_color_and_gray(path)
            if bgr is None:
                logger.warning(f"Warning: Unable to load image at {path}.")
                continue

            # Hasler & Süsstrunk colorfulness
            b, g, r = cv2.split(bgr.astype(np.float32))
            rg = r - g
            yb = 0.5 * (r + g) - b
            std_rg, std_yb = np.std(rg), np.std(yb)
            mean_rg, mean_yb = np.mean(rg), np.mean(yb)
            colorfulness = float(
                np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
            )

            # Saturation stats in HSV
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1].astype(np.float32) / 255.0
            mean_sat = float(np.mean(s))
            low_sat_pct = float(np.count_nonzero(s < 0.10) / s.size * 100.0)
            high_sat_pct = float(np.count_nonzero(s > 0.90) / s.size * 100.0)

            results.append(
                {
                    "name": os.path.basename(path),
                    "colorfulness": colorfulness,
                    "mean_saturation": mean_sat,
                    "low_sat_pct": low_sat_pct,
                    "high_sat_pct": high_sat_pct,
                }
            )

            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(images_path), kwargs.get("progress_callback", None)
            )

        data = pd.DataFrame(results)
        self.serializer.save_images_colorimetry(data)
        logger.info(f"Color metrics computed: {len(data)} records")
        return data

    # -------------------------------------------------------------------------
    def calculate_edge_metrics(self, images_path: list[str], **kwargs) -> pd.DataFrame:
        results: list[dict] = []
        for i, path in enumerate(
            tqdm(images_path, desc="Edge metrics", total=len(images_path), ncols=100)
        ):
            _, gray = self.load_color_and_gray(path)
            if gray is None:
                logger.warning(f"Warning: Unable to load image at {path}.")
                continue

            # Auto Canny thresholds based on median
            v = float(np.median(gray))
            lower = int(max(0, (1.0 - 0.33) * v))
            upper = int(min(255, (1.0 + 0.33) * v))
            edges = cv2.Canny(gray, lower, upper)
            edge_density = float(np.count_nonzero(edges) / edges.size)

            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mean_gradient = float(np.mean(np.hypot(gx, gy)))

            results.append(
                {
                    "name": os.path.basename(path),
                    "edge_density": edge_density,
                    "mean_gradient": mean_gradient,
                }
            )

            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(images_path), kwargs.get("progress_callback", None)
            )

        data = pd.DataFrame(results)
        self.serializer.save_images_edges_metrics(data)
        logger.info(f"Edge metrics computed: {len(data)} records")
        return data

    # -------------------------------------------------------------------------
    def calculate_texture_lbp_metrics(
        self, images_path: list[str], **kwargs
    ) -> pd.DataFrame:
        def compute_lbp_8u1(gray_u8: np.ndarray) -> np.ndarray:
            # Use vectorized 8-neighbour comparisons (ignore 1-pixel border)
            gray_f = gray_u8.astype(np.uint8)
            center = gray_f[1:-1, 1:-1]

            weights = [1, 2, 4, 8, 16, 32, 64, 128]  # clockwise
            n = []
            n.append(gray_f[0:-2, 0:-2])  # top-left
            n.append(gray_f[0:-2, 1:-1])  # top
            n.append(gray_f[0:-2, 2:])  # top-right
            n.append(gray_f[1:-1, 2:])  # right
            n.append(gray_f[2:, 2:])  # bottom-right
            n.append(gray_f[2:, 1:-1])  # bottom
            n.append(gray_f[2:, 0:-2])  # bottom-left
            n.append(gray_f[1:-1, 0:-2])  # left

            lbp = np.zeros_like(center, dtype=np.uint16)
            for w, neigh in zip(weights, n):
                lbp |= (neigh >= center).astype(np.uint16) * w
            return lbp  # shape (H-2, W-2), values 0..255

        results: list[dict] = []
        for i, path in enumerate(
            tqdm(
                images_path,
                desc="Texture (LBP) metrics",
                total=len(images_path),
                ncols=100,
            )
        ):
            _, gray = self.load_color_and_gray(path)
            if gray is None:
                logger.warning(f"Warning: Unable to load image at {path}.")
                continue

            lbp = compute_lbp_8u1(gray)
            hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
            p = hist.astype(np.float64)
            p /= p.sum() + 1e-12
            lbp_energy = float(np.sum(p * p))
            lbp_entropy = float(-np.sum(np.where(p > 0, p * np.log2(p), 0.0)))

            results.append(
                {
                    "name": os.path.basename(path),
                    "lbp_energy": lbp_energy,
                    "lbp_entropy": lbp_entropy,
                }
            )

            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i + 1, len(images_path), kwargs.get("progress_callback", None)
            )

        data = pd.DataFrame(results)
        self.serializer.save_images_texture_metric(data)
        logger.info(f"Texture (LBP) metrics computed: {len(data)} records")
        return data
