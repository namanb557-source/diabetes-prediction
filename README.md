# Diabetes Prediction System (ML + Flask)

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run `pip install -r requirements.txt`.
4. Run `python app.py`.
5. Open http://127.0.0.1:5000

The app downloads a public Pima Indians diabetes teaching dataset at startup and trains a Logistic Regression pipeline. Internet access is required on first startup / deployment.

## Deploy for a public URL (Render)
1. Create a GitHub account and a new public repository.
2. Upload all files and folders from this project.
3. Sign in to Render and choose **New + → Web Service**.
4. Connect your GitHub repository.
5. Render should detect `render.yaml`. If asked, use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
6. Deploy. Render will provide a public `https://...onrender.com` URL that can be opened on another laptop.

The free service may sleep when unused and take a short time to wake up.

## Academic / medical note
This is an educational ML demonstration, not a diagnostic tool. It uses the Pima Indians dataset, which is small and population-specific. Its predictions and probabilities are not medically validated and must not guide healthcare decisions.
