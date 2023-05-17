from madewithml.predict import get_best_run_id
from madewithml.serve import CustomLogic, Model


best_run_id = get_best_run_id('llm', 'val_loss', 'ASC')
CustomLogic.bind(model=Model.bind(run_id=best_run_id), threshold=0.9)
