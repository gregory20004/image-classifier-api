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

Примечание: файл весов модели (resnet_model_cifar.pt) не включён в репозиторий из-за размера. 
Для запуска обучи модель самостоятельно или запроси у автора.
