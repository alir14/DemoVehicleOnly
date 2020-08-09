#include "LicensePlateReader.hpp"

LicensePlateReader::LicensePlateReader(InferenceEngine::Core& ie,
	const std::string& xmlPath,
	const std::map<std::string, std::string>& pluginConfig):
	ie_{ie}
{
	auto network = ie.ReadNetwork(xmlPath);

	InferenceEngine::InputsDataMap lprInputInfo(network.getInputsInfo());

	if (lprInputInfo.size() != 1 && lprInputInfo.size() != 2)
	{
		throw std::logic_error("LPR should have 1 or 2 inputs");
	}

	InferenceEngine::InputInfo::Ptr& lprInputInfoFirst = lprInputInfo.begin()->second;
	lprInputInfoFirst->setPrecision(InferenceEngine::Precision::U8);
	lprInputInfoFirst->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
	lprInputInfoFirst->setLayout(InferenceEngine::Layout::NHWC);

	LprInputName = lprInputInfo.begin()->first;

	if (lprInputInfo.size() == 2)
	{
		auto sequenceInput = (++lprInputInfo.begin());
		LprInputSeqName = sequenceInput->first;
	}
	else
	{
		LprInputSeqName = "";
	}

	InferenceEngine::OutputsDataMap lprOutputInfo(network.getOutputsInfo());
	if (lprOutputInfo.size() != 1)
	{
		throw std::logic_error("LPR should have 1 output");
	}

	LprOutputName = lprOutputInfo.begin()->first;
	size_t indexOfSequenceSize = LprOutputName == "" ? 2 : 1;

	maxSequenceSizePerPlate = lprOutputInfo.begin()->second->getTensorDesc().getDims()[indexOfSequenceSize];

	net = ie_.LoadNetwork(network, "CPU", pluginConfig);

	std::cout << "initialize lpr ..." << std::endl;
}

InferenceEngine::InferRequest LicensePlateReader::createInferRequest()
{
	return net.CreateInferRequest();
}

void LicensePlateReader::setImage(InferenceEngine::InferRequest& inferRequest, 
	const cv::Mat& img, 
	const cv::Rect plateRect)
{
	InferenceEngine::Blob::Ptr roiBlob = inferRequest.GetBlob(LprInputName);
	
	if(InferenceEngine::Layout::NHWC == roiBlob->getTensorDesc().getLayout())
	{
		InferenceEngine::ROI cropRoi{
			0,
			static_cast<size_t>(plateRect.x),
			static_cast<size_t>(plateRect.y),
			static_cast<size_t>(plateRect.width),
			static_cast<size_t>(plateRect.height)
		};
		InferenceEngine::Blob::Ptr frameblob = wrapMat2Blob(img);
		InferenceEngine::Blob::Ptr roiBlob = make_shared_blob(frameblob, cropRoi);
		inferRequest.SetBlob(LprInputName, roiBlob);
	}
	else
	{
		std::cout << "non nhwc" << std::endl;
		const cv::Mat& vehicleImage = img(plateRect);
		matU8ToBlob<uint8_t>(vehicleImage, roiBlob);
	}

	if (LprInputSeqName != "") 
	{
		InferenceEngine::Blob::Ptr seqBlob = inferRequest.GetBlob(LprInputSeqName);
		float* blob_data = seqBlob->buffer().as<float*>();
		blob_data[0] = 0.0f;
		std::fill(blob_data + 1, blob_data + seqBlob->getTensorDesc().getDims()[0], 1.0f);
	}
}

std::string LicensePlateReader::getResult(InferenceEngine::InferRequest& inferRequest)
{
	std::cout << "plate result " << std::endl;

	static const char* const items[] = {
		"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
		"", "", "", "",
		"", "", "", "",
		"", "", "", "",
		"", "", "", "",
		"", "", "", "",
		"", "", "", "",
		"", "", "", "",
		"", "", "", "",
		"", "",
		"A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
		"K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
		"U", "V", "W", "X", "Y", "Z"
	};

	std::string result;
	result.reserve(14u + 6u);
	const auto data = inferRequest.GetBlob(LprOutputName)->buffer().as<float*>();
	for (size_t i = 0; i < maxSequenceSizePerPlate; i++)
	{
		if (data[i] == -1)
			break;

		result += items[std::size_t(data[i])];
	}

	return result;
}

std::string LicensePlateReader::ProcessAndReadPalteNumber(InferenceEngine::InferRequest& inferRequest, 
	const cv::Mat& img, 
	const cv::Rect plateRect)
{
	try
	{
		cv::Mat currentFrame = img.clone();
		std::cout << "plate setiamge " << std::endl;
		setImage(inferRequest, currentFrame, plateRect);
		std::cout << "plate infre " << std::endl;
		inferRequest.Infer();
		std::cout << "plate get result " << std::endl;
		return getResult(inferRequest);
	}
	catch (const std::exception& ex)
	{
		std::cout << ex.what() << std::endl;
	}
	return "";
}