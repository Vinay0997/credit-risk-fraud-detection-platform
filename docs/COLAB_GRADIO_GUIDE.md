# Google Colab Gradio Guide

This guide launches the Gradio web interface from Google Colab. Gradio can create a public `gradio.live` link directly from the notebook.

## 1. Clone the GitHub Repository

Replace the repository URL if your GitHub repo name is different.

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

## 3. Launch the Gradio App

Run this cell and keep it running:

```python
!GRADIO_SHARE=true python app/gradio_app.py
```

Gradio will print a public link like:

```text
Running on public URL: https://something.gradio.live
```

Open that `gradio.live` link in your browser.

## 4. Use the App

Use the app in this order:

1. Open `Overview` and click `Generate Dataset`.
2. Open `Model Scoring` and click `Train Models`.
3. Click `Score Portfolio`.
4. Open `Explainability` and click `Explain Record`.
5. Open `Monitoring` and click `Run Drift Report`.
6. Open `Policy Assistant` and ask a compliance or model-risk question.
7. Open `Downloads` and click `Prepare Downloads`.

## 5. Common Questions

If you do not see the link, wait a few seconds and scroll to the bottom of the running Colab cell.

If the app stops, rerun:

```python
!GRADIO_SHARE=true python app/gradio_app.py
```

If you changed the GitHub repo and Colab still shows the old app, rerun the clone cell with `rm -rf` so Colab pulls a fresh copy.
