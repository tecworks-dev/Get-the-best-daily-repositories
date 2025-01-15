### **🦆 DuckDB Quick Starter para Usuários de PostgreSQL**  

Se você já conhece **PostgreSQL**, pode usar **DuckDB** diretamente no terminal sem precisar de configuração.  
Aqui estão os principais comandos **SQL** para rodar no **DuckDB CLI**.

---

## **1️⃣ Criar e Usar um Banco de Dados**
```sql
-- Criar um banco de dados chamado "meudb.duckdb"
ATTACH 'meudb.duckdb' AS meudb;
```
Isso cria um **arquivo de banco de dados** persistente.

---

## **2️⃣ Criar uma Tabela**
```sql
CREATE TABLE meudb.usuarios (
    id INTEGER PRIMARY KEY,
    nome TEXT,
    email TEXT UNIQUE,
    idade INTEGER,
    criado_em TIMESTAMP DEFAULT now()
);
```
**📝 Observações:**  
✅ DuckDB usa `TEXT` em vez de `VARCHAR`.  
✅ `TIMESTAMP DEFAULT now()` adiciona a data e hora automaticamente.  

---

## **3️⃣ Inserir Dados**
```sql
INSERT INTO meudb.usuarios (id, nome, email, idade) VALUES
    (1, 'Alice', 'alice@email.com', 25),
    (2, 'Bob', 'bob@email.com', 30),
    (3, 'Carol', 'carol@email.com', 22);
```

✅ **No DuckDB, você pode inserir múltiplas linhas de uma vez!** 🚀  

---

## **4️⃣ Consultar os Dados**
```sql
SELECT * FROM meudb.usuarios;
```

🔹 **Filtrar por idade:**  
```sql
SELECT nome, idade FROM meudb.usuarios WHERE idade > 25;
```

🔹 **Ordenar resultados:**  
```sql
SELECT * FROM meudb.usuarios ORDER BY idade DESC;
```

🔹 **Contar registros:**  
```sql
SELECT COUNT(*) FROM meudb.usuarios;
```

---

## **5️⃣ Atualizar e Deletar Dados**
```sql
-- Atualizar a idade de Bob
UPDATE meudb.usuarios SET idade = 35 WHERE nome = 'Bob';

-- Deletar um usuário
DELETE FROM meudb.usuarios WHERE nome = 'Carol';
```

---

## **6️⃣ Trabalhando com Agregações**
🔹 **Média de idade dos usuários:**  
```sql
SELECT AVG(idade) AS idade_media FROM meudb.usuarios;
```

🔹 **Quantidade de usuários por idade:**  
```sql
SELECT idade, COUNT(*) FROM meudb.usuarios GROUP BY idade;
```

🔹 **Maior e menor idade:**  
```sql
SELECT MIN(idade) AS menor_idade, MAX(idade) AS maior_idade FROM meudb.usuarios;
```

---

## **7️⃣ Trabalhando com Datas**
🔹 **Ver usuários cadastrados nos últimos 7 dias:**  
```sql
SELECT * FROM meudb.usuarios WHERE criado_em > now() - INTERVAL '7 days';
```

🔹 **Formatar data:**  
```sql
SELECT nome, STRFTIME('%Y-%m-%d', criado_em) AS data_criacao FROM meudb.usuarios;
```

---

## **8️⃣ Exportar e Importar Dados**
🔹 **Salvar dados em CSV:**  
```sql
COPY meudb.usuarios TO 'usuarios.csv' WITH (HEADER, DELIMITER ',');
```

🔹 **Carregar CSV para o DuckDB:**  
```sql
CREATE TABLE meudb.novos_usuarios AS SELECT * FROM read_csv_auto('usuarios.csv');
```

🔹 **Salvando arquivos Parquet diretamente:**  
```sql
COPY meudb.usuarios TO 'usuarios.parquet' (FORMAT 'parquet');
```

```sql
CREATE TABLE meudb.novos_usuarios AS SELECT * FROM read_parquet('usuarios.parquet');
```

🔹 **Salvando arquivos JSON diretamente:**  
```sql
COPY meudb.usuarios TO 'usuarios.json' (FORMAT 'json');
```

```sql
CREATE TABLE meudb.novos_usuarios AS SELECT * FROM read_json('usuarios.json');
```

Para **ler um banco de dados DuckDB** e visualizar suas tabelas, siga os comandos abaixo.

---

## **1️⃣ Conectar ao Banco DuckDB**
Se você salvou o banco de dados em um arquivo (ex: `meudb.duckdb`), primeiro **anexe** o banco ao DuckDB:
```sql
ATTACH 'meudb.duckdb' AS meudb;
```
Isso carrega o banco de dados **persistente**.

Se quiser apenas usar o banco **em memória**, ignore esse passo.

---

## **2️⃣ Ver Todas as Tabelas no Banco**
Para listar todas as tabelas existentes no banco DuckDB:
```sql
SHOW TABLES;
```
Isso mostrará todas as tabelas disponíveis no banco **atual**.

Se o banco está **anexado** (`ATTACH`), e você quer listar as tabelas dentro dele, rode:
```sql
SHOW TABLES FROM meudb;
```

---

## **3️⃣ Ver Estrutura de uma Tabela**
Se quiser verificar a estrutura (schema) de uma tabela específica:
```sql
DESCRIBE meudb.usuarios;
```

Ou, para obter mais detalhes:
```sql
PRAGMA table_info('meudb.usuarios');
```

---

## **4️⃣ Consultar os Dados de uma Tabela**
Para visualizar os dados de uma tabela:
```sql
SELECT * FROM meudb.usuarios LIMIT 10;
```

Se quiser contar quantos registros existem:
```sql
SELECT COUNT(*) FROM meudb.usuarios;
```

---

## **🔥 Conclusão**
| **Comando** | **Descrição** |
|------------|-------------|
| `ATTACH 'meudb.duckdb' AS meudb;` | Conectar um banco de dados DuckDB |
| `DESCRIBE meudb.usuarios;` | Mostrar estrutura da tabela |
| `PRAGMA table_info('meudb.usuarios');` | Mostrar detalhes da tabela |
| `SELECT * FROM meudb.usuarios LIMIT 10;` | Consultar os primeiros registros |
| `SELECT COUNT(*) FROM meudb.usuarios;` | Contar registros da tabela |

Agora você pode **ler e explorar qualquer banco DuckDB**! 🚀🔥

---

## **🔥 Conclusão**
Agora você pode usar **DuckDB como um mini PostgreSQL local**, sem precisar de servidor! 🚀  
Se precisar de mais comandos ou quiser testar funções avançadas, me avise! 🔥🔥