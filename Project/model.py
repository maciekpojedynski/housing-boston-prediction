from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from .preprocessing import build_preprocessor

def build_model():
    preprocessor = build_preprocessor()
    model = make_pipeline(preprocessor, LinearRegression())
    return model