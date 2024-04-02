# Deep Learn Framework
> by **KingYen.**  
> form **QIT Software Studio**

## The project tree
```markdown
.
├── README.md
├── __pycache__
│   ├── function.cpython-311.pyc
│   ├── utils.cpython-311.pyc
│   └── variable.cpython-311.pyc
├── environment.yaml
├── function.py
├── main.py
├── tests
│   ├── __pycache__
│   │   └── test_square.cpython-311.pyc
│   └── test_square.py
├── utils.py
└── variable.py

```

## Get Start
### Environment preparation
1. Make sure you have configured your 'Anaconda' environment
2. run it
    ```shell
    conda env create -f environment.yaml
    ```
3. Write the main file, such as this file
    ```python
    from function import square, exp
    from variable import Variable

    x = Variable(0.5)
    y = square(exp(square(x)))
    y.backward()
    print(x.grad)
    ```
4. Run main.go
    ```shell
   python main.py
    ```
5. Run the unit test
    ```shell
    python -m unittest discover tests 
    ```

