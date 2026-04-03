# Image Classifier API

FastAPI-сервис для классификации изображений на базе ResNet18 (CIFAR-10).

## Стек
Python, FastAPI, PyTorch, Docker, Pydantic

## Запуск
docker-compose up

## Эндпоинты
POST /predict — классификация изображения
GET /health — статус сервиса  
GET /info — информация о модели
