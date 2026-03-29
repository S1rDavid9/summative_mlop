"""
locustfile.py — Load testing for the Student Engagement Detection API
----------------------------------------------------------------------

Tests the POST /predict endpoint by sending synthetic 48×48 grayscale
face images at scale.

How to run:
-----------
1. Install Locust:
       pip install locust

2. Headless (command-line) mode — recommended for CI / automated tests:
       locust -f tests/locustfile.py \
              --host=http://localhost:8000 \
              --headless \
              --users 100 \
              --spawn-rate 10 \
              --run-time 60s

3. Web UI mode (open http://localhost:8089 in your browser):
       locust -f tests/locustfile.py --host=http://localhost:8000

Suggested test scenarios (change --users accordingly):
    10  users  —  light load
    50  users  —  moderate load
    100 users  —  heavy load
    500 users  —  stress test

Expected results table (fill in after running):
-----------------------------------------------
| Users | Req/s | Median (ms) | 95th % (ms) | Failures |
|-------|-------|-------------|-------------|----------|
|    10 |       |             |             |          |
|    50 |       |             |             |          |
|   100 |       |             |             |          |
|   500 |       |             |             |          |
"""

import io
import random
import numpy as np

from PIL import Image
from locust import HttpUser, task, between, events


# ── Shared synthetic image bytes (generated once at module load) ─────────────

def _make_synthetic_image_bytes(size: int = 48) -> bytes:
    """
    Generate a synthetic 48×48 grayscale PNG in memory.
    We add a tiny random noise so each request payload is slightly different,
    which better simulates real traffic.
    """
    base = np.full((size, size), 128, dtype=np.uint8)
    noise = np.random.randint(-20, 20, (size, size), dtype=np.int16)
    img_array = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L').convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


# Pre-generate a small pool of synthetic images so workers don't all hit the
# RNG at the same time (reduces CPU overhead during heavy load tests).
_IMAGE_POOL = [_make_synthetic_image_bytes() for _ in range(20)]


# ── Locust user class ────────────────────────────────────────────────────────

class EmotionAPIUser(HttpUser):
    """
    Simulates a client that sends face images to the /predict endpoint.

    wait_time: each virtual user waits 1–3 seconds between requests,
    approximating a real application that processes one frame at a time.
    """
    wait_time = between(1, 3)

    # ── Tasks ────────────────────────────────────────────────────────────────

    @task(10)
    def predict_emotion(self):
        """
        POST /predict  — highest-frequency task (weight=10).

        Picks a random image from the pre-generated pool and POSTs it.
        The response is validated to confirm the API returned a prediction.
        """
        img_bytes = random.choice(_IMAGE_POOL)
        files = {'file': ('face.png', img_bytes, 'image/png')}

        with self.client.post(
            '/predict',
            files=files,
            name='/predict',
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'predicted_class' not in data:
                    response.failure(
                        f'Missing predicted_class in response: {data}'
                    )
                else:
                    response.success()
            elif response.status_code == 503:
                # Model not loaded — mark as failure but don't crash
                response.failure('Model not loaded (503)')
            else:
                response.failure(
                    f'Unexpected status {response.status_code}: '
                    f'{response.text[:200]}'
                )

    @task(1)
    def health_check(self):
        """
        GET /health  — low-frequency background polling (weight=1).

        Simulates the UI Monitor page pinging the health endpoint.
        """
        with self.client.get(
            '/health',
            name='/health',
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f'Health check failed: {response.status_code}')

    @task(1)
    def get_metrics(self):
        """
        GET /metrics  — low-frequency metrics polling (weight=1).
        """
        with self.client.get(
            '/metrics',
            name='/metrics',
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f'Metrics failed: {response.status_code}')

    # ── Lifecycle hooks ──────────────────────────────────────────────────────

    def on_start(self):
        """Called once when a virtual user starts."""
        pass

    def on_stop(self):
        """Called once when a virtual user stops."""
        pass


# ── Event hooks (optional: log summary to stdout) ────────────────────────────

@events.quitting.add_listener
def on_locust_quit(environment, **kwargs):
    stats = environment.stats
    total = stats.total
    print('\n' + '='*55)
    print('  LOAD TEST SUMMARY')
    print('='*55)
    print(f'  Total requests   : {total.num_requests}')
    print(f'  Failures         : {total.num_failures}')
    if total.num_requests > 0:
        fail_pct = 100 * total.num_failures / total.num_requests
        print(f'  Failure rate     : {fail_pct:.1f}%')
    print(f'  Median resp (ms) : {total.median_response_time}')
    print(f'  95th %ile (ms)   : {total.get_response_time_percentile(0.95)}')
    print(f'  Req/s            : {total.total_rps:.2f}')
    print('='*55 + '\n')
