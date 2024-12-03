# 环境配置
要求是torch1.7.1，cuda>10.1
结合服务器有的cuda，以及torch版本匹配的cuda，选择在GPU上的cuda10.2
在.zshrc修改cuda路径，source .zshrc

```bash
conda env create -f ./env/mid_env.yaml
```

