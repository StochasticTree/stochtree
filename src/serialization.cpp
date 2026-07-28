#include <cpp11.hpp>
#include "stochtree_types.h"
#include <stochtree/container.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/random_effects.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>
#include <nlohmann/json.hpp>

[[cpp11::register]]
cpp11::external_pointer<nlohmann::json> init_json_cpp() {
    std::unique_ptr<nlohmann::json> json_ptr = std::make_unique<nlohmann::json>();
    json forests = nlohmann::json::object();
    json rfx = nlohmann::json::object();
    json parameters = nlohmann::json::object();
    json_ptr->emplace("forests", forests);
    json_ptr->emplace("random_effects", rfx);
    json_ptr->emplace("num_forests", 0);
    json_ptr->emplace("num_random_effects", 0);
    return cpp11::external_pointer<nlohmann::json>(json_ptr.release());
}

[[cpp11::register]]
void json_add_double_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name, double field_value) {
    if (json_ptr->contains(subfolder_name)) {
        if (json_ptr->at(subfolder_name).contains(field_name)) {
            json_ptr->at(subfolder_name).at(field_name) = field_value;
        } else {
            json_ptr->at(subfolder_name).emplace(std::pair(field_name, field_value));
        }
    } else {
        json_ptr->emplace(std::pair(subfolder_name, nlohmann::json::object()));
        json_ptr->at(subfolder_name).emplace(std::pair(field_name, field_value));
    }
}

[[cpp11::register]]
void json_add_double_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name, double field_value) {
    if (json_ptr->contains(field_name)) {
        json_ptr->at(field_name) = field_value;
    } else {
        json_ptr->emplace(std::pair(field_name, field_value));
    }
}

[[cpp11::register]]
void json_add_integer_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name, int field_value) {
    if (json_ptr->contains(subfolder_name)) {
        if (json_ptr->at(subfolder_name).contains(field_name)) {
            json_ptr->at(subfolder_name).at(field_name) = field_value;
        } else {
            json_ptr->at(subfolder_name).emplace(std::pair(field_name, field_value));
        }
    } else {
        json_ptr->emplace(std::pair(subfolder_name, nlohmann::json::object()));
        json_ptr->at(subfolder_name).emplace(std::pair(field_name, field_value));
    }
}

[[cpp11::register]]
void json_add_integer_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name, int field_value) {
    if (json_ptr->contains(field_name)) {
        json_ptr->at(field_name) = field_value;
    } else {
        json_ptr->emplace(std::pair(field_name, field_value));
    }
}

[[cpp11::register]]
void json_add_bool_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name, bool field_value) {
    if (json_ptr->contains(subfolder_name)) {
        if (json_ptr->at(subfolder_name).contains(field_name)) {
            json_ptr->at(subfolder_name).at(field_name) = field_value;
        } else {
            json_ptr->at(subfolder_name).emplace(std::pair(field_name, field_value));
        }
    } else {
        json_ptr->emplace(std::pair(subfolder_name, nlohmann::json::object()));
        json_ptr->at(subfolder_name).emplace(std::pair(field_name, field_value));
    }
}

[[cpp11::register]]
void json_add_bool_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name, bool field_value) {
    if (json_ptr->contains(field_name)) {
        json_ptr->at(field_name) = field_value;
    } else {
        json_ptr->emplace(std::pair(field_name, field_value));
    }
}

[[cpp11::register]]
void json_add_vector_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name, cpp11::doubles field_vector) {
    int vec_length = field_vector.size();
    if (json_ptr->contains(subfolder_name)) {
        if (json_ptr->at(subfolder_name).contains(field_name)) {
            json_ptr->at(subfolder_name).at(field_name).clear();
            for (int i = 0; i < vec_length; i++) {
                json_ptr->at(subfolder_name).at(field_name).emplace_back(field_vector.at(i));
            }
        } else {
            json_ptr->at(subfolder_name).emplace(std::pair(field_name, nlohmann::json::array()));
            for (int i = 0; i < vec_length; i++) {
                json_ptr->at(subfolder_name).at(field_name).emplace_back(field_vector.at(i));
            }
        }
    } else {
        json_ptr->emplace(std::pair(subfolder_name, nlohmann::json::object()));
        json_ptr->at(subfolder_name).emplace(std::pair(field_name, nlohmann::json::array()));
        for (int i = 0; i < vec_length; i++) {
            json_ptr->at(subfolder_name).at(field_name).emplace_back(field_vector.at(i));
        }
    }
}

[[cpp11::register]]
void json_add_vector_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name, cpp11::doubles field_vector) {
    int vec_length = field_vector.size();
    if (json_ptr->contains(field_name)) {
        json_ptr->at(field_name).clear();
        for (int i = 0; i < vec_length; i++) {
            json_ptr->at(field_name).emplace_back(field_vector.at(i));
        }
    } else {
        json_ptr->emplace(std::pair(field_name, nlohmann::json::array()));
        for (int i = 0; i < vec_length; i++) {
            json_ptr->at(field_name).emplace_back(field_vector.at(i));
        }
    }
}

[[cpp11::register]]
void json_add_integer_vector_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name, cpp11::integers field_vector) {
    int vec_length = field_vector.size();
    if (json_ptr->contains(subfolder_name)) {
        if (json_ptr->at(subfolder_name).contains(field_name)) {
            json_ptr->at(subfolder_name).at(field_name).clear();
            for (int i = 0; i < vec_length; i++) {
                json_ptr->at(subfolder_name).at(field_name).emplace_back(field_vector.at(i));
            }
        } else {
            json_ptr->at(subfolder_name).emplace(std::pair(field_name, nlohmann::json::array()));
            for (int i = 0; i < vec_length; i++) {
                json_ptr->at(subfolder_name).at(field_name).emplace_back(field_vector.at(i));
            }
        }
    } else {
        json_ptr->emplace(std::pair(subfolder_name, nlohmann::json::object()));
        json_ptr->at(subfolder_name).emplace(std::pair(field_name, nlohmann::json::array()));
        for (int i = 0; i < vec_length; i++) {
            json_ptr->at(subfolder_name).at(field_name).emplace_back(field_vector.at(i));
        }
    }
}

[[cpp11::register]]
void json_add_integer_vector_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name, cpp11::integers field_vector) {
    int vec_length = field_vector.size();
    if (json_ptr->contains(field_name)) {
        json_ptr->at(field_name).clear();
        for (int i = 0; i < vec_length; i++) {
            json_ptr->at(field_name).emplace_back(field_vector.at(i));
        }
    } else {
        json_ptr->emplace(std::pair(field_name, nlohmann::json::array()));
        for (int i = 0; i < vec_length; i++) {
            json_ptr->at(field_name).emplace_back(field_vector.at(i));
        }
    }
}

[[cpp11::register]]
void json_add_string_vector_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name, cpp11::strings field_vector) {
    int vec_length = field_vector.size();
    if (json_ptr->contains(subfolder_name)) {
        if (json_ptr->at(subfolder_name).contains(field_name)) {
            json_ptr->at(subfolder_name).at(field_name).clear();
            for (int i = 0; i < vec_length; i++) {
                json_ptr->at(subfolder_name).at(field_name).emplace_back(field_vector.at(i));
            }
        } else {
            json_ptr->at(subfolder_name).emplace(std::pair(field_name, nlohmann::json::array()));
            for (int i = 0; i < vec_length; i++) {
                json_ptr->at(subfolder_name).at(field_name).emplace_back(field_vector.at(i));
            }
        }
    } else {
        json_ptr->emplace(std::pair(subfolder_name, nlohmann::json::object()));
        json_ptr->at(subfolder_name).emplace(std::pair(field_name, nlohmann::json::array()));
        for (int i = 0; i < vec_length; i++) {
            json_ptr->at(subfolder_name).at(field_name).emplace_back(field_vector.at(i));
        }
    }
}

[[cpp11::register]]
void json_add_string_vector_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name, cpp11::strings field_vector) {
    int vec_length = field_vector.size();
    if (json_ptr->contains(field_name)) {
        json_ptr->at(field_name).clear();
        for (int i = 0; i < vec_length; i++) {
            json_ptr->at(field_name).emplace_back(field_vector.at(i));
        }
    } else {
        json_ptr->emplace(std::pair(field_name, nlohmann::json::array()));
        for (int i = 0; i < vec_length; i++) {
            json_ptr->at(field_name).emplace_back(field_vector.at(i));
        }
    }
}

[[cpp11::register]]
void json_add_string_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name, std::string field_value) {
    if (json_ptr->contains(subfolder_name)) {
        if (json_ptr->at(subfolder_name).contains(field_name)) {
            json_ptr->at(subfolder_name).at(field_name) = field_value;
        } else {
            json_ptr->at(subfolder_name).emplace(std::pair(field_name, field_value));
        }
    } else {
        json_ptr->emplace(std::pair(subfolder_name, nlohmann::json::object()));
        json_ptr->at(subfolder_name).emplace(std::pair(field_name, field_value));
    }
}

[[cpp11::register]]
void json_add_string_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name, std::string field_value) {
    if (json_ptr->contains(field_name)) {
        json_ptr->at(field_name) = field_value;
    } else {
        json_ptr->emplace(std::pair(field_name, field_value));
    }
}

[[cpp11::register]]
bool json_contains_field_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name) {
    if (json_ptr->contains(subfolder_name)) {
        if (json_ptr->at(subfolder_name).contains(field_name)) {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

[[cpp11::register]]
bool json_contains_field_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name) {
    if (json_ptr->contains(field_name)) {
        return true;
    } else {
        return false;
    }
}

[[cpp11::register]]
double json_extract_double_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name) {
    return json_ptr->at(subfolder_name).at(field_name);
}

[[cpp11::register]]
double json_extract_double_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name) {
    return json_ptr->at(field_name);
}

[[cpp11::register]]
int json_extract_integer_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name) {
    return json_ptr->at(subfolder_name).at(field_name);
}

[[cpp11::register]]
int json_extract_integer_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name) {
    return json_ptr->at(field_name);
}

[[cpp11::register]]
bool json_extract_bool_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name) {
    return json_ptr->at(subfolder_name).at(field_name);
}

[[cpp11::register]]
bool json_extract_bool_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name) {
    return json_ptr->at(field_name);
}

[[cpp11::register]]
std::string json_extract_string_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name) {
    return json_ptr->at(subfolder_name).at(field_name);
}

[[cpp11::register]]
std::string json_extract_string_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name) {
    return json_ptr->at(field_name);
}

[[cpp11::register]]
cpp11::writable::doubles json_extract_vector_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name) {
    cpp11::writable::doubles output;
    int vec_length = json_ptr->at(subfolder_name).at(field_name).size();
    for (int i = 0; i < vec_length; i++) output.push_back((json_ptr->at(subfolder_name).at(field_name).at(i)));
    return output;
}

[[cpp11::register]]
cpp11::writable::doubles json_extract_vector_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name) {
    cpp11::writable::doubles output;
    int vec_length = json_ptr->at(field_name).size();
    for (int i = 0; i < vec_length; i++) output.push_back((json_ptr->at(field_name).at(i)));
    return output;
}

[[cpp11::register]]
cpp11::writable::integers json_extract_integer_vector_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name) {
    cpp11::writable::integers output;
    int vec_length = json_ptr->at(subfolder_name).at(field_name).size();
    for (int i = 0; i < vec_length; i++) output.push_back((json_ptr->at(subfolder_name).at(field_name).at(i)));
    return output;
}

[[cpp11::register]]
cpp11::writable::integers json_extract_integer_vector_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name) {
    cpp11::writable::integers output;
    int vec_length = json_ptr->at(field_name).size();
    for (int i = 0; i < vec_length; i++) output.push_back((json_ptr->at(field_name).at(i)));
    return output;
}

[[cpp11::register]]
cpp11::writable::strings json_extract_string_vector_subfolder_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string subfolder_name, std::string field_name) {
    int vec_length = json_ptr->at(subfolder_name).at(field_name).size();
    std::vector<std::string> output(vec_length);
    for (int i = 0; i < vec_length; i++) output.at(i) = json_ptr->at(subfolder_name).at(field_name).at(i);
    return output;
}

[[cpp11::register]]
cpp11::writable::strings json_extract_string_vector_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string field_name) {
    int vec_length = json_ptr->at(field_name).size();
    std::vector<std::string> output(vec_length);
    for (int i = 0; i < vec_length; i++) output.at(i) = json_ptr->at(field_name).at(i);
    return output;
}

[[cpp11::register]]
std::string json_add_forest_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, cpp11::external_pointer<StochTree::ForestContainer> forest_samples) {
    int forest_num = json_ptr->at("num_forests");
    std::string forest_label = "forest_" + std::to_string(forest_num);
    nlohmann::json forest_json = forest_samples->to_json();
    json_ptr->at("forests").emplace(forest_label, forest_json);
    json_ptr->at("num_forests") = forest_num + 1;
    return forest_label;
}

[[cpp11::register]]
void json_increment_rfx_count_cpp(cpp11::external_pointer<nlohmann::json> json_ptr) {
    int rfx_num = json_ptr->at("num_random_effects");
    json_ptr->at("num_random_effects") = rfx_num + 1;
}

[[cpp11::register]]
std::string json_add_rfx_container_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, cpp11::external_pointer<StochTree::RandomEffectsContainer> rfx_samples) {
    int rfx_num = json_ptr->at("num_random_effects");
    std::string rfx_label = "random_effect_container_" + std::to_string(rfx_num);
    nlohmann::json rfx_json = rfx_samples->to_json();
    json_ptr->at("random_effects").emplace(rfx_label, rfx_json);
    return rfx_label;
}

[[cpp11::register]]
std::string json_add_rfx_label_mapper_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, cpp11::external_pointer<StochTree::LabelMapper> label_mapper) {
    int rfx_num = json_ptr->at("num_random_effects");
    std::string rfx_label = "random_effect_label_mapper_" + std::to_string(rfx_num);
    nlohmann::json rfx_json = label_mapper->to_json();
    json_ptr->at("random_effects").emplace(rfx_label, rfx_json);
    return rfx_label;
}

[[cpp11::register]]
std::string json_add_rfx_groupids_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, cpp11::integers groupids) {
    int rfx_num = json_ptr->at("num_random_effects");
    std::string rfx_label = "random_effect_groupids_" + std::to_string(rfx_num);
    nlohmann::json groupids_json = nlohmann::json::array();
    for (int i = 0; i < groupids.size(); i++) {
        groupids_json.emplace_back(groupids.at(i));
    }
    json_ptr->at("random_effects").emplace(rfx_label, groupids_json);
    return rfx_label;
}

[[cpp11::register]]
std::string get_json_string_cpp(cpp11::external_pointer<nlohmann::json> json_ptr) {
    return json_ptr->dump();
}

[[cpp11::register]]
void json_save_file_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string filename) {
    std::ofstream output_file(filename);
    output_file << *json_ptr << std::endl;
}

[[cpp11::register]]
void json_load_file_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string filename) {
    std::ifstream f(filename);
    // nlohmann::json file_json = nlohmann::json::parse(f);
    *json_ptr = nlohmann::json::parse(f);
    // json_ptr.reset(&file_json);
}

[[cpp11::register]]
void json_load_string_cpp(cpp11::external_pointer<nlohmann::json> json_ptr, std::string json_string) {
    *json_ptr = nlohmann::json::parse(json_string);
}
