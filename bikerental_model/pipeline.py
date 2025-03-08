import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikerental_model.config.core import config
from bikerental_model.processing.features import WeekdayImputer
from bikerental_model.processing.features import WeathersitImputer
from bikerental_model.processing.features import Mapper, OutlierHandler, WeekdayOneHotEncoder
from bikerental_model.processing.features import Mapper

bikerental_pipe=Pipeline([
    
    ("weekday_imputer", WeekdayImputer()
     ),
    ("weatherisit_imputer", WeathersitImputer()
     ),
    ("weekday_mapping", Mapper(config.model_config_.weekday_var, config.model_config_.weekday_mapping)),
    ("workingday_mapping", Mapper(config.model_config_.workingday_var, config.model_config_.workingday_mapping)),
    ("weathersit_mapping", Mapper(config.model_config_.weathersit_var, config.model_config_.weathersit_mapping)),
    ("season_mapping", Mapper(config.model_config_.season_var, config.model_config_.season_mapping)),
    ("hr_mapping", Mapper(config.model_config_.hr_var, config.model_config_.hr_mapping)),
    ("holiday_mappings", Mapper(config.model_config_.holiday_var, config.model_config_.holiday_mappings)),
    ("outlier_handler", OutlierHandler()),
    ("weekday_one_hot_encoder", WeekdayOneHotEncoder(variable="weekday")),
    ('regressor',RandomForestRegressor())
     ])
