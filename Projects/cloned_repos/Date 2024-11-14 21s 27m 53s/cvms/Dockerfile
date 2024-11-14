# 1st stage, build app
FROM golang:1.23 AS builder

WORKDIR /build

COPY go.mod go.sum ./

RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags "-s -w" -trimpath -o ./cvms ./cmd/cvms/main.go

# 2nd stage, copy CA certificates
FROM alpine:latest AS certs
RUN apk --no-cache add ca-certificates

# 3rd stage, run app
FROM scratch AS production

WORKDIR /var/lib/cvms

COPY --from=builder /build/cvms /bin/cvms

COPY --from=certs /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

ENTRYPOINT ["/bin/cvms"]
