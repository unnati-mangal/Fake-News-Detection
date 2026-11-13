FAKE NEWS DETECTOR - STREAMLIT APP
==================================

FILES
  app.py                  the web app
  requirements.txt        libraries needed
  run.bat / run.sh        one-click start (Windows / Mac-Linux)
  fake_news_full.ipynb    the full training notebook (run in Colab)
  fake_news_pipeline.pkl  <-- YOU MUST ADD THIS FILE (see step 1)

STEP 1: Get the model file
  Run fake_news_full.ipynb in Colab (needs Fake.csv and True.csv).
  After section 12 it creates fake_news_pipeline.pkl.
  Download it from the Colab Files panel and put it in THIS folder,
  next to app.py.

STEP 2: Install Python 3.9+ (if not installed) from python.org

STEP 3: Run
  Windows : double-click run.bat
  Mac/Linux: run  ./run.sh
  Or manually:
      pip install -r requirements.txt
      streamlit run app.py
  The app opens at http://localhost:8501

TROUBLESHOOTING
  - "fake_news_pipeline.pkl not found": the .pkl is not in the same folder as app.py.
  - Error while loading the .pkl / version warning: your local scikit-learn
    version differs from Colab. In Colab run:  import sklearn; print(sklearn.__version__)
    then locally run:  pip install scikit-learn==THAT_VERSION
  - "streamlit is not recognized": use  python -m streamlit run app.py

DEPLOY FREE (optional)
  Push this folder (including the .pkl) to GitHub, then deploy on
  share.streamlit.io. Pin the scikit-learn version in requirements.txt.

NOTE: trained on US political news from 2016-2017; results on other topics
or years may be unreliable. Classroom project, not a fact-checker.
