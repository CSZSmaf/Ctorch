#include <dl/dl.h>

int main() {
    auto value = dl::ones({2, 3}, {.requires_grad = true});
    dl::sum(value * value).backward();
    return value.grad()->to_vector<float>() == std::vector<float>(6, 2.f)
        && dl::describe(value).find("shape=[2, 3]") != std::string::npos ? 0 : 1;
}
