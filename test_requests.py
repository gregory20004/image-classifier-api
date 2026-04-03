import requests

# отправка картинки как файла
with open("cat.jpg", "rb") as f :
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("cat.jpg", f, "image/jpeg")}
    )
print(response.json())

health = requests.get(
    "http://localhost:8000/health",
)
print(health.json())

info = requests.get(
    "http://localhost:8000/info",
)
print(info.json())
