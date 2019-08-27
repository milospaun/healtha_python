#CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(190) NOT NULL,
    lastname VARCHAR(190) NOT NULL,
    email VARCHAR(190) UNIQUE NOT NULL,
    password VARCHAR(190) NOT NULL,
    role VARCHAR(190),
    role_id INTEGER,
    created_at TIMESTAMP
)

#CREATE TABLE IF NOT EXISTS data(
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    gx DOUBLE(10,3),
    gy DOUBLE(10,3),
    gz DOUBLE(10,3),
    ax DOUBLE(10,3),
    ay DOUBLE(10,3),
    az DOUBLE(10,3),
    timestamp TIMESTAMP,
    user_id INTEGER,
    source ENUM('ios', 'android', 'watch'),
    created_at TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
)

#CREATE TABLE IF NOT EXISTS statistics(
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(190) NOT NULL,
    hours DOUBLE(20, 4) NOT NULL DEFAULT 0.0
    user_id INTEGER
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
)