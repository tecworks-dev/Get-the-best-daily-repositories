This document explains how to start server (@Pollens/server) locally with Docker

> **Warning**:
>
> This document is not guaranteed to be up-to-date.
> If you find any outdated information, please feel free to open an issue or submit a PR.

## Run postgresql in docker

```
docker pull postgres
docker run --rm --name Pollens-postgres -e POSTGRES_PASSWORD=Pollens -p 5432:5432 -v ~/Documents/postgres:/var/lib/postgresql/data postgres
```

### Optionally, use a dedicated volume

```
docker volume create Pollens-postgres
docker run --rm --name Pollens-postgres -e POSTGRES_PASSWORD=Pollens -p 5432:5432 -v Pollens-postgres:/var/lib/postgresql/data postgres
```

### mailhog (for local testing)

```
docker run --rm --name mailhog -p 1025:1025 -p 8025:8025 mailhog/mailhog
```

## prepare db

```
docker ps
docker exec -it CONTAINER_ID psql -U postgres ## change container_id
```

### in the terminal, following the example to user & table

```
psql (15.3 (Debian 15.3-1.pgdg120+1))
Type "help" for help.

postgres=# CREATE USER Pollens WITH PASSWORD 'Pollens';
CREATE ROLE
postgres=# ALTER USER Pollens WITH SUPERUSER;
ALTER ROLE
postgres=# CREATE DATABASE Pollens;
CREATE DATABASE
postgres=# \du
                                   List of roles
 Role name |                         Attributes                         | Member of
-----------+------------------------------------------------------------+-----------
 Pollens    | Superuser                                                  | {}
 postgres  | Superuser, Create role, Create DB, Replication, Bypass RLS | {}
```

### Set the following config to `packages/backend/server/.env`

In the following setup, we assume you have postgres server running at localhost:5432 and mailhog running at localhost:1025.

When logging in via email, you will see the mail arriving at localhost:8025 in a browser.

```
DATABASE_URL="postgresql://Pollens:Pollens@localhost:5432/Pollens"
MAILER_SENDER="noreply@PollensAI.info"
MAILER_USER="auth"
MAILER_PASSWORD="auth"
MAILER_HOST="localhost"
MAILER_PORT="1025"
```

## Prepare prisma

```
yarn workspace @Pollens/server prisma db push
yarn workspace @Pollens/server data-migration run
```

Note, you may need to do it again if db schema changed.

### Enable prisma studio

```
yarn workspace @Pollens/server prisma studio
```

## Build native packages (you need to setup rust toolchain first)

```
# build native
yarn workspace @Pollens/server-native build
yarn workspace @Pollens/native build
```

## start server

```
yarn workspace @Pollens/server dev
```

## start core (web)

```
yarn dev
```

## Done

Now you should be able to start developing Pollens with server enabled.
