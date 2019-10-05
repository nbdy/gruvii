## gruvii
[![Build Status](http://build.eberlein.io:8080/job/gruvii/badge/icon)](http://build.eberlein.io:8080/job/gruvii/)<br>

[GRUV](https://github.com/smthnspcl/GRUV) v2
### why
original version was depended on keras 0.1.0<br>
this uses tensorflow v2
### from cli
```bash
./gruvii.py --help

usage: gruvii.py {arguments}
{arguments}		{default value}
	--help
	-d	--dataset-directory	./dataset/test/
	-i	--iterations		50
	-s	--sampling-frequency	44100
	-c	--clip-length		10
	-h	--hidden-dimensions	1024
	-b	--batch-size		5
	-e	--epochs		25
```

### from script
```python
from gruvii import Trainer, Configuration

t = Trainer(Configuration.parse())
prefix = t.prepare_data()
t.train(prefix)
```