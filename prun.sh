py="python3.7 ./main.py"
is_running(){
    pgrep -f "$1" > /dev/null
}

while true; do
    if ! is_running "$py"; then
    echo "restarting..."
    $py &
    fi
    sleep 60
done
