
#include <atomic>

template<typename T>
class LFQueue
{
private:
    /* data */
    std::atomic<uint32_t> count{0};
    std::atomic<uint32_t> tail{0};
    alignas(64)T internal_buffer[4096*8];
    T* extension_buffer;
public:
    LFQueue(/* args */);
    ~LFQueue();
    void push_back(T value);
    T get_elem(uint16_t i);
    uint16_t get_count();
};
template<typename T>
LFQueue<T>::LFQueue(/* args */)
{
}
template<typename T>
LFQueue<T>::~LFQueue()
{
}
template<typename T>
void LFQueue<T>::push_back(T value){
    uint32_t count_read = count.fetch_add(1,std::memory_order_acquire);
    if (count_read > 4096*8){
        count.fetch_sub(1,std::memory_order_acquire);
        return;
    }
    uint32_t acquired_tail = tail.fetch_add(1,std::memory_order_acquire);
    internal_buffer[acquired_tail] = value;
}

template<typename T>
T LFQueue<T>::get_elem(uint16_t i) {
    return internal_buffer[i];
}

template<typename T>
uint16_t LFQueue<T>::get_count() {
    return count.load(std::memory_order_relaxed);
}
