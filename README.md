# Movie Recommendation System

## Overview
This project implements a movie recommendation system using different recommendation strategies, including content-based filtering and popularity-based filtering. It utilizes movie ratings and metadata to provide personalized recommendations.

## Features
- Content-based filtering: Recommends movies based on movie metadata similarity.
- Popularity-based filtering: Recommends top-rated movies based on user ratings.
- Data-driven approach using CSV datasets.

## Installation
### Prerequisites
Ensure you have Python installed (preferably Python 3.7 or later). Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
### Running the Project
Run the main script to generate recommendations:
```bash
python main.py
```

### Data Files
- `movies.csv`: Contains movie metadata.
- `ratings.csv`: Contains user ratings for movies.

## Project Structure
```
Recommandation_System_Project/
│-- content_based.py  # Content-based recommendation script
│-- popularity_based.py  # Popularity-based recommendation script
│-- main.py  # Main entry point
│-- movies.csv  # Movie dataset
│-- ratings.csv  # Ratings dataset
│-- requirements.txt  # Dependencies
```

## Contributing
Feel free to fork this repository and contribute improvements! Submit a pull request with a clear description of changes.

## License
This project is open-source and available under the MIT License.

