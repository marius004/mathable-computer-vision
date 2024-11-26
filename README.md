# Mathable Algorithm

## Documentation

Here you can find the [documentation](./DOCUMENTATION.md)

## Requirements

Here you can find the
[requirements](./requirements.txt) or run the following script in your venv:

```bash
pip install -r requirements.txt
```

## Running the notebook

To run all tasks, you only need to make two simple adjustments:

* Update the range for the game loop

    In the function predict, ensure that the range is set to ```nr_of_tests + 1```:

    ```py
    def predict(input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        for game in range(1, 5): # Update this range to match your test count
    ```

* Modify the ```input_dir``` and ```output_dir``` in the last cell of the notebook

    Open the [mathable.ipynb](mathable.ipynb) notebook and update the last cell to specify your input and output directories. For example:

    ```py
    predict(
        input_dir='evaluare/fake_test',
        output_dir='evaluare/fisiere_solutie/352_Scarlat_Marius'
    )
    ```