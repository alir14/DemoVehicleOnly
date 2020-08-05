#pragma once

#include <string>

const bool auto_Resize_FLAG = true;
const bool FLAGS_h = false;
const std::string FLAGS_i = "";
std::string vehicleLicense_Model = "D:\\workspace\\openvino\\models\\intel\\vehicle-license-plate-detection-barrier-0106\\FP16\\vehicle-license-plate-detection-barrier-0106.xml";
std::string vehicleAttribute_Model = "D:\\workspace\\openvino\\models\\intel\\vehicle-attributes-recognition-barrier-0039\\FP16\\vehicle-attributes-recognition-barrier-0039.xml";
std::string plateLicense_Model = "D:\\workspace\\openvino\\models\\intel\\license-plate-recognition-barrier-0001\\FP16\\license-plate-recognition-barrier-0001.xml";

std::string vehicle_device = "CPU";
const std::string vehicle_Attribute_device = "CPU";
const std::string plate_Licese_device = "CPU";
const double threshold_flag = 0.5;
const bool FLAGS_no_show = false;
const bool FLAGS_auto_resize = false;
const bool FLAGS_loop_video = false;
const uint32_t numberOfallocatedFrame = 3;
const uint32_t nunber_workerThread = 1;
const uint32_t FLAGS_nthreads = 0;
