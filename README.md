# Learning Biostatistics with Python 🐍

Welcome to my repository for learning biostatistics! This project serves as a personal log and toolkit as I work through my biostatistics course, translating concepts and assignments into a working knowledge of Python. The goal is to build a strong, open-source foundation in data analysis that will be useful for future academic and professional work. This repository documents my learning process and contains reusable tools for common statistical tasks.

## Repository Structure

The project is organized to keep learning materials, datasets, and custom tools separate and easy to navigate. You can also find markdown files that explain how to perform specific tasks, such as generating practice data.

Biostats.ipynb: This is the main Jupyter Notebook where I perform analyses, visualize data, and document my work.

datasets/: Any datasets used in the notebooks are stored here.

my_stats_tools.py: A custom Python module containing reusable functions for common statistical tasks, such as creating stem-and-leaf plots or frequency distributions.

creating_practice_data.md: This file explains how to create datasets on which to practice statistical operations.

library_references.md: This file lists a variety of interesting Python libraries that may be worth exploring.

## 🚀 Getting Started

Follow these steps to set up your environment and explore the project.

1. **Prerequisites:** I started using the [Anaconda Distribution](https://www.anaconda.com/download), which comes pre-packaged with Python and all the necessary data science libraries and other goodies (pandas, NumPy, Matplotlib, Seaborn, Jupyter, etc.). While not necessarily a prerequisite, I did find it useful to install almost everything in one go.

2. **Clone the Repository:** Clone this repository to your local machine using Git

```Python
git clone https://github.com/A-Napodano/public-health-notebook.git
cd public-health-notebook
```

3. **Launch Jupyter Notebook**

```Python
jupyter notebook
```

## 🛠️ Using the Custom Tools

The my_stats_tools.py file contains helper functions to simplify common analyses. To use these functions in any notebook, simply place the notebook in the root directory or add the project path, then import the module at the top of your notebook.

```python
import my_stats_tools as mst
```

Feel free to edit or add your own functions to this file as you build out your toolkit!
