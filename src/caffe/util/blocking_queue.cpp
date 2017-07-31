#include <boost/thread.hpp>
#include <string>

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template<typename T>
class BlockingQueue<T>::sync {
 public:
  mutable boost::mutex mutex_;           // 可变锁
  boost::condition_variable condition_;  // 条件变量
};

template<typename T> 
BlockingQueue<T>::BlockingQueue()        // 构造函数, 新建个同步对象
    : sync_(new sync()) {
}

template<typename T>
void BlockingQueue<T>::push(const T& t) {        // 锁, push, 解锁, 条件通知
  boost::mutex::scoped_lock lock(sync_->mutex_);
  queue_.push(t);
  lock.unlock();
  sync_->condition_.notify_one();
}

template<typename T>
bool BlockingQueue<T>::try_pop(T* t) {        // 空了就返回false, 不空, 弹出
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
T BlockingQueue<T>::pop(const string& log_on_wait) { // 空了就等待, 把等待的时长输出, 返回类型, 强制弹出
  boost::mutex::scoped_lock lock(sync_->mutex_);

  while (queue_.empty()) {
    if (!log_on_wait.empty()) {
      LOG_EVERY_N(INFO, 1000)<< log_on_wait;
    }
    sync_->condition_.wait(lock); 
  }

  T t = queue_.front();
  queue_.pop();
  return t;
}

template<typename T>
bool BlockingQueue<T>::try_peek(T* t) {  // 空了返回, 不空是返回的指针, 
  boost::mutex::scoped_lock lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  return true;
}

template<typename T>
T BlockingQueue<T>::peek() {      // 空了等待, 返回指针
  boost::mutex::scoped_lock lock(sync_->mutex_);

  while (queue_.empty()) {
    sync_->condition_.wait(lock);
  }

  return queue_.front();
}

template<typename T>
size_t BlockingQueue<T>::size() const {  // 
  boost::mutex::scoped_lock lock(sync_->mutex_);
  return queue_.size();
}

template class BlockingQueue<Batch<float>*>;
template class BlockingQueue<Batch<double>*>;

}  // namespace caffe
