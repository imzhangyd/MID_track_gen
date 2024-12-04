## 环境配置
要求是torch1.7.1，cuda>10.1
结合服务器有的cuda，以及torch版本匹配的cuda，选择在GPU上的cuda10.2
在.zshrc修改cuda路径，source .zshrc

```bash
conda env create -f ./env/mid_env.yaml
```

.whl文件可以手动下载，直接 pip install 就行

## 数据集格式
txt格式存储的dataframe, 四列 
```
frame，trackid，x, y
```

## 先跑一次原数据的pipe

