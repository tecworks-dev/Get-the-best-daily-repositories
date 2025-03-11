# Ollama 负载均衡服务器

Ollama 负载均衡服务器是一款高性能、易配置的开源负载均衡服务器，优化Ollama负载。它能够帮助您提高应用程序的可用性和响应速度，同时确保系统资源的有效利用。

## 特性

- **高性能** - 采用先进的算法和技术来实现高效的请求分发。
- **易配置** - 简单直观的配置文件使得部署和调整变得轻而易举。
- **可扩展性** - 支持动态添加或移除后端服务节点，无需重启服务器。
- **安全性** - 无原有Ollama漏洞利用路径相关路由，无法通过该负载均衡服务器删除源服务器模型及数据。

## 使用场景

- 提升Web应用和服务的可用性
- 均衡分布式系统的负载
- 快速故障转移和恢复
- 作为代理服务器，不使原生ollama端口对外暴露

## 说明

`ollama.db`: 存储源服务器列表，通过`/api/tags`接口获取，可使用自建Ollama或公网开放的Ollama服务器

## 支持接口

`/`: Ollama is running

`/api/tags`: 已启用的模型列表

`/v1/models`: 已启用的模型列表(OpenAI接口)

`/api/version`: Ollama版本号

`/api/chat`：LLM模型调用

`/api/embed`：embedding模型调用

`/api/embeddings`：embedding模型调用


不支持`/api/show`、`/api/copy`、`/api/create`、`/api/push`、`/api/delete`、`/api/pull`、`/api/ps`等有一定危险性的接口

