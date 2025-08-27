from fastapi import FastAPI
 
# Initialise FastAPI
app = FastAPI()
 
# Define a test route
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is working!"}