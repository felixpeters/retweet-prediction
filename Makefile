build-image:
	docker build -t felixpeters/retweet-prediction .

update-image: build-image
	docker push felixpeters/retweet-prediction:latest
