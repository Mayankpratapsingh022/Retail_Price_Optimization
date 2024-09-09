FROM bentoml/model-server:latest
COPY ./model /service/model
CMD ["bentoml", "serve", "/service/model"]
