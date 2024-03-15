# Quarter 2 Project

The project described here represents an innovative approach to analyzing the degradation of perovskite solar films through data processing and machine learning, leveraging the experimental data from the Solar Energy Innovation Laboratory (SOLEIL). This document aims to provide a comprehensive understanding of the project's goals and methodology.

### Project Overview ###

Our project centers on the utilization of experimental data collected by SOLEIL, focusing on the manufacturing processes and materials used in the production of perovskite solar films. Additionally, the project incorporates a unique dataset of GoPro images taken at regular intervals to document the degradation of these films over time. The primary objective is to develop a more streamlined and quantifiable method of tracking film degradation, moving away from manual assessments to a data-driven model.

Two main type of data are used in the project:
- Manufacturing Data: Detailed records of the manufacturing process and materials used for each perovskite solar film
- GoPro Images: A series of images captured at set intervals, documenting the visual degradation of the solar films over time.
    
The project employs an innovative approach to process the GoPro images. For each image, we calculate an average black value for each film, termed "colormetrics." This method allows us to represent the degradation process through a single numerical value per film, facilitating a more straightforward analysis and comparison.

Utilizing the colormetrics derived from the GoPro images alongside the manufacturing data, we deploy machine learning models to predict the degradation path of each solar film. The aim is to identify patterns and correlations between the manufacturing processes/materials used and the rate at which the films degrade. By doing so, we hope to uncover insights that could lead to the development of more durable perovskite solar films.

This project represents a pivotal step toward harnessing machine learning for the analysis of solar film degradation. By transforming visual degradation indicators into quantifiable data, we pave the way for more sophisticated analyses and predictive modeling in the field of solar energy research. Future iterations of the project will aim to incorporate a larger dataset of GoPro images, refine the machine learning models, and explore additional variables that may impact film degradation.

## How to Run

To run this project follow these instructions:
- in a console navigate to the folder containing the project files.
- in the console run ```pip install -r requirements.txt``` to ensure that you have all the necessary packages to run the project.
- within Jupyter Notebooks you can run the 'GDBInterface and CurveParamPredictor Guide' notebook which demonstrates an example workflow using the Python functions and classes within the 'root' folder.
To run the colormetrics guide:
- Download the colormetrics_guide.ipynb Jupyter notebook, which demonstrates an example workflow using the Python functions and classes within the 'root' folder!
