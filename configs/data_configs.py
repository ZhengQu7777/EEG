def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        # External CV datasets carry their source/session in the name, while
        # sharing the same binary resting-state input contract.
        if str(dataset_name).startswith((
            "Stroke_HC",
            "stroke_hc_eeg_fnir_epo_data",
        )):
            return all
        # The TUEG tasks share the same two-class signal contract, but use
        # 19-channel clinical EEG rather than the older 16-channel dataset.
        if str(dataset_name).startswith("TUEG_"):
            return tueg_stroke
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

# all的裁剪的数据集
class all():
    def __init__(self):
        super(all, self).__init__()
        self.num_classes = 2
        self.sequence_len = 2000
        self.class_names = ['HC', 'Stroke']
        self.input_channels = 16

# closeeye的裁剪的数据集
class closeeye():
    def __init__(self):
        super(closeeye, self).__init__()
        self.num_classes = 2
        self.sequence_len = 2000
        self.class_names = ['HC', 'Stroke']
        self.input_channels = 16

# openeye的裁剪的数据集
class openeye():
    def __init__(self):
        super(openeye, self).__init__()
        self.num_classes = 2
        self.sequence_len = 2000
        self.class_names = ['HC', 'Stroke']
        self.input_channels = 16


class tueg_stroke():
    def __init__(self):
        super(tueg_stroke, self).__init__()
        self.num_classes = 2
        self.sequence_len = 2000
        self.class_names = ['ClinicallyNormal', 'Stroke']
        self.input_channels = 19


class stroke():
    def __init__(self):
        super(stroke, self).__init__()
        self.num_classes = 3
        self.sequence_len = 2000
        self.class_names = ['HC', 'IS', 'HS']
        self.input_channels = 19


class ishs():
    """Ischaemic-versus-haemorrhagic stroke view of the HCISHS cohort."""
    def __init__(self):
        super(ishs, self).__init__()
        self.num_classes = 2
        self.sequence_len = 2000
        self.class_names = ['IS', 'HS']
        self.input_channels = 19
