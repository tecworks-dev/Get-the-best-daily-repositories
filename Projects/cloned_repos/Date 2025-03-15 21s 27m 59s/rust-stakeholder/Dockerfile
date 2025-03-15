FROM rust

WORKDIR /app
COPY ./Cargo.* /app
COPY ./src/ /app/src
RUN cargo build --release
ENTRYPOINT [ "/app/target/release/rust-stakeholder" ]