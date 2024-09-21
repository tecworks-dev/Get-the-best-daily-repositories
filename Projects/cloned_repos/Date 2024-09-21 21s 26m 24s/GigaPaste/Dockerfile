FROM alpine:latest

RUN apk add --no-cache git go

WORKDIR /app

COPY . /app

ENV CGO_ENABLED=1

RUN go build -ldflags="-s -w"

CMD ["./app"]
