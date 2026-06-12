# Google Colab Streamlit Guide

This guide launches the Streamlit interface from Google Colab using Cloudflare Tunnel. Cloudflare is more reliable than LocalTunnel for Streamlit because Streamlit loads JavaScript and CSS files dynamically.

## 1. Clone the GitHub Repository

Replace the repository URL if your GitHub repo name is different.

```python
!git clone https://github.com/Vinay0997/credit-risk-fraud-detection-platform.git
%cd credit-risk-fraud-detection-platform
```

If the folder already exists in Colab, use:

```python
%cd /content
!rm -rf credit-risk-fraud-detection-platform
!git clone https://github.com/Vinay0997/credit-risk-fraud-detection-platform.git
%cd credit-risk-fraud-detection-platform
```

## 2. Install Dependencies

```python
!pip install -q -r requirements.txt
!pip install -q -e .
```

If Colab asks you to restart the runtime, restart it and rerun the clone/install cells.

## 3. Start Streamlit

```python
!pkill -f streamlit || true

!streamlit run app/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  > streamlit.log 2>&1 &
```

Check the Streamlit log:

```python
!sleep 5
!cat streamlit.log
```

## 4. Create the Live Web Link

```python
!wget -q -O cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared
!./cloudflared tunnel --url http://localhost:8501
```

Open the URL that looks like this:

```text
https://something.trycloudflare.com
```

Keep this Colab cell running while you use the Streamlit app.

## 5. Use the App

In the sidebar:

1. Click `Generate Dataset`.
2. Click `Train Models`.
3. Click `Score Portfolio`.

Then use the tabs:

- `Overview`: synthetic portfolio and signal summary.
- `Model Scoring`: model metrics, scored records, and explanations.
- `Monitoring`: drift report simulation.
- `Policy Assistant`: citation-backed policy Q&A.
- `Downloads`: export dataset, scores, metrics, and drift report.

## Troubleshooting

If the app page does not load:

```python
!cat streamlit.log
```

If the tunnel stops, rerun:

```python
!./cloudflared tunnel --url http://localhost:8501
```

If you used LocalTunnel and saw errors like `Failed to fetch dynamically imported module`, stop LocalTunnel and use Cloudflare Tunnel instead.
