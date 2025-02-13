DOCKER_DEFAULT_PLATFORM=linux/amd64 docker buildx build --platform=linux/amd64 -t us-east4-docker.pkg.dev/gen-lang-client-0834671745/whisper-serving/whisper-serving:latest .

docker push us-east4-docker.pkg.dev/gen-lang-client-0834671745/whisper-serving/whisper-serving:latest

gcloud run deploy my-whisper-service --image=us-east4-docker.pkg.dev/gen-lang-client-0834671745/whisper-serving/whisper-serving:latest --region=us-east4 --project=gen-lang-client-0834671745 --service-account=genai-slackbot-dev@gen-lang-client-0834671745.iam.gserviceaccount.com

