#include "rubik/coco.hpp"

#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace rubik {

using nlohmann::json;

std::unordered_map<int, const CocoImage*> CocoDataset::images_by_id() const {
    std::unordered_map<int, const CocoImage*> out;
    out.reserve(images.size());
    for (const auto& im : images) out[im.id] = &im;
    return out;
}

std::unordered_map<int, std::string> CocoDataset::category_name_by_id() const {
    std::unordered_map<int, std::string> out;
    out.reserve(categories.size());
    for (const auto& c : categories) out[c.id] = c.name;
    return out;
}

std::optional<int> CocoDataset::category_id_by_name(const std::string& name) const {
    for (const auto& c : categories) {
        if (c.name == name) return c.id;
    }
    return std::nullopt;
}

CocoDataset load_coco(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Could not open COCO json: " + path.string());
    json j;
    in >> j;

    CocoDataset coco;
    if (j.contains("info") && j["info"].is_object() && j["info"].contains("description")) {
        coco.description = j["info"]["description"].get<std::string>();
    }

    for (const auto& jc : j.value("categories", json::array())) {
        CocoCategory c;
        c.id = jc.value("id", 0);
        c.name = jc.value("name", "");
        c.supercategory = jc.value("supercategory", std::string("object"));
        if (jc.contains("keypoints")) {
            for (const auto& kp : jc["keypoints"]) c.keypoints.push_back(kp.get<std::string>());
        }
        if (jc.contains("skeleton")) {
            for (const auto& sk : jc["skeleton"]) {
                c.skeleton.push_back({sk.at(0).get<int>(), sk.at(1).get<int>()});
            }
        }
        coco.categories.push_back(std::move(c));
    }

    for (const auto& ji : j.value("images", json::array())) {
        CocoImage im;
        im.id = ji.value("id", 0);
        im.file_name = ji.value("file_name", "");
        im.width = ji.value("width", 0);
        im.height = ji.value("height", 0);
        coco.images.push_back(std::move(im));
    }

    for (const auto& ja : j.value("annotations", json::array())) {
        CocoAnnotation ann;
        ann.id = ja.value("id", 0);
        ann.image_id = ja.value("image_id", 0);
        ann.category_id = ja.value("category_id", 0);
        if (ja.contains("bbox")) {
            const auto& b = ja["bbox"];
            for (size_t i = 0; i < 4 && i < b.size(); ++i) ann.bbox[i] = b[i].get<double>();
        }
        ann.area = ja.value("area", 0.0);
        ann.iscrowd = ja.value("iscrowd", 0);
        if (ja.contains("num_keypoints")) ann.num_keypoints = ja["num_keypoints"].get<int>();
        if (ja.contains("keypoints")) {
            for (const auto& v : ja["keypoints"]) ann.keypoints.push_back(v.get<double>());
        }
        coco.annotations.push_back(std::move(ann));
    }

    return coco;
}

void save_coco(const CocoDataset& coco, const std::filesystem::path& path, bool minimal_images) {
    json j;
    j["info"] = {{"description", coco.description}, {"version", "1.0"}};
    j["licenses"] = minimal_images ? json::array() : json::array({{{"name", ""}, {"id", 0}, {"url", ""}}});

    json jcats = json::array();
    for (const auto& c : coco.categories) {
        json jc;
        jc["id"] = c.id;
        jc["name"] = c.name;
        jc["supercategory"] = c.supercategory;
        jc["keypoints"] = c.keypoints;
        json jskel = json::array();
        for (const auto& sk : c.skeleton) jskel.push_back(json::array({sk[0], sk[1]}));
        jc["skeleton"] = jskel;
        jcats.push_back(std::move(jc));
    }
    j["categories"] = std::move(jcats);

    json jimages = json::array();
    for (const auto& im : coco.images) {
        if (minimal_images) {
            jimages.push_back(
                {{"id", im.id}, {"file_name", im.file_name}, {"width", im.width}, {"height", im.height}});
        } else {
            jimages.push_back({{"id", im.id},
                                {"file_name", im.file_name},
                                {"width", im.width},
                                {"height", im.height},
                                {"license", 0},
                                {"flickr_url", ""},
                                {"coco_url", ""},
                                {"date_captured", 0}});
        }
    }
    j["images"] = std::move(jimages);

    json janns = json::array();
    for (const auto& ann : coco.annotations) {
        json ja;
        ja["id"] = ann.id;
        ja["image_id"] = ann.image_id;
        ja["category_id"] = ann.category_id;
        ja["bbox"] = json::array({ann.bbox[0], ann.bbox[1], ann.bbox[2], ann.bbox[3]});
        ja["area"] = ann.area;
        ja["iscrowd"] = ann.iscrowd;
        if (ann.num_keypoints >= 0) ja["num_keypoints"] = ann.num_keypoints;
        if (!ann.keypoints.empty()) ja["keypoints"] = ann.keypoints;
        janns.push_back(std::move(ja));
    }
    j["annotations"] = std::move(janns);

    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out << j.dump();
}

std::optional<std::filesystem::path> resolve_image(const std::filesystem::path& images_root,
                                                     const std::string& file_name) {
    namespace fs = std::filesystem;
    fs::path direct = images_root / file_name;
    if (fs::exists(direct)) return direct;

    fs::path basename = fs::path(file_name).filename();
    fs::path flat = images_root / basename;
    if (fs::exists(flat)) return flat;

    if (!fs::exists(images_root)) return std::nullopt;

    std::vector<fs::path> matches;
    for (const auto& entry : fs::recursive_directory_iterator(images_root)) {
        if (entry.is_regular_file() && entry.path().filename() == basename) {
            matches.push_back(entry.path());
        }
    }
    if (matches.size() == 1) return matches.front();
    return std::nullopt;
}

}  // namespace rubik
