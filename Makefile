build-image:
	docker build -t felixpeters/retweet-prediction .

update-image: build-image
	docker push felixpeters/retweet-prediction:latest

build-gpu-image:
	docker build -t felixpeters/retweet-prediction-gpu -f Dockerfile.gpu .

update-gpu-image: build-gpu-image
	docker push felixpeters/retweet-prediction-gpu:latest
