FROM rust:slim

RUN \
        apt update && \
        apt install -y git

WORKDIR /app

COPY Cargo.toml .
COPY Cargo.lock .
RUN mkdir src
RUN touch src/lib.rs

RUN cargo fetch

COPY src .

RUN cargo test --no-run

CMD [ "cargo", "test" ]
