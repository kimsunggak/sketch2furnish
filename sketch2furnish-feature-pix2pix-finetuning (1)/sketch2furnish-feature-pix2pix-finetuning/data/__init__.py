"""이 패키지는 데이터셋을 불러오고 전처리하는 데 필요한 여러 모듈들을 포함함

 새로운 데이터셋을 사용하고 싶다면, 이 패키지 내에 있는 틀(template)을 참고하여 직접 코드를 작성할 수 있다
 예를 들어, "dummy"라는 새로운 데이터셋을 사용하고 싶다면 dummy_dataset.py라는 파일을 만들고, 그 안에 DummyDataset이라는 클래스를 정의하면 된다 
 이때, DummyDataset 클래스는 이미 제공된 BaseDataset이라는 기본 클래스(Base class)를 상속받아야 한다

    -- __init__(self, opt): 클래스 초기화 함수, opt는 여러 옵션(예: 데이터 경로, 배치 사이즈 등)을 담은 객체
    -- __len__(self):   데이터셋에 들어있는 데이터(예: 이미지)의 총 개수를 반환
    -- __getitem__(self, index):                   get a data point from data loader.
    -- <modify_commandline_options>:(optionally) 데이터셋이 특별한 전처리 옵션이나 설정이 필요하다면 이 함수를 구현

커맨드라인에서 --dataset_mode dummy라는 옵션을 주면 자동으로 사용됨 
이미 template_dataset.py라는 템플릿 파일이 제공되므로, 이를 참고하여 새로운 데이터셋을 만들 수 있다
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    그 파일 안에 있는 클래스 중에서 [DatasetName]Dataset 클래스를 찾아서 반환
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def create_dataset(opt):
    """
    학습 또는 테스트 스크립트(예: train.py, test.py)에서 호출되어, 옵션(opt)에 따라 올바른 데이터셋을 생성
    전체 데이터 로딩 파이프라인의 진입점 역할을 함
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """ 실제 데이터셋 객체를 PyTorch DataLoader로 감싸, 멀티스레딩을 통해 데이터를 효율적으로 로드할 수 있도록 함"""
    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """만약 데이터셋 크기가 옵션에 지정된 max_dataset_size보다 크면, 그 값을 제한하여 반환함"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            # 현재까지 처리한 데이터 샘플 수가 max_dataset_size보다 크면 중단
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
