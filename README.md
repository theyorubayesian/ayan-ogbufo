# Project Setup
- Create a virtual environment and activate it. Use Python 3.8 only.
```
$ py -3.8 -m venv env
```
- Clone the project
```
$ git clone https://github.com/theyorubayesian/ayan-ogbufo.git
```
- Install requirements
```
$ pip install -r requirements.txt
```
- Install torch==1.6.0. For Windows, no CUDA, use the command below:
```
$ pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
- Install joeynmt in bash
```
./install-joeynmt
```
- Read [Collaboration Notes](#collaboration-notes) 
- Open `explore.ipynb` and hack away! 🔨🔨

# Collaboration Notes
- Always work out of a new branch.
- Create new tasks as issues. It helps!
- Always add docstrings to functions and classes.
- Use Typing to specify argument and return types. 
- One import per line. Use absolute imports everywhere possible.
- Squash your commits into one before pushing to remote.

# Resources
List papers that inform any decisions here.
- [Revisiting Low-Resource Neural Machine Translation: A Case Study (Sennrich et al, 2019)](https://www.aclweb.org/anthology/P19-1021/)
