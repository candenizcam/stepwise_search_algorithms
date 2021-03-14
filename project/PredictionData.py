class PredictionData:
    def __init__(self, tr_class, tr_preds, te_class, te_preds):
        """
        This is a data class to ease access to the data when using for training and predictions
        :param tr_class: training classification row
        :param tr_preds: training predictors
        :param te_class: testing classification row
        :param te_preds: testing predictors
        """
        self.tr_class = tr_class
        self.tr_preds = tr_preds
        self.te_class = te_class
        self.te_preds = te_preds