import datetime
from datetime import timezone

def format_timestamp(ts):
    """Formats a timestamp (UNIX timestamp or datetime object) into a human-readable local time string."""
    if not ts:
        return "N/A"
    
    try:
        # Handle datetime objects (from database)
        if isinstance(ts, datetime.datetime):
            # If datetime is naive (no timezone), assume it's UTC
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            # Convert to local time
            local_dt = ts.astimezone()
            return local_dt.strftime('%m/%d/%Y, %I:%M:%S %p')
        
        # Handle UNIX timestamps (int/float)
        elif isinstance(ts, (int, float)):
            dt = datetime.datetime.fromtimestamp(ts)
            return dt.strftime('%m/%d/%Y, %I:%M:%S %p')
        
        else:
            return "N/A"
            
    except (ValueError, TypeError, OSError):
        return "Invalid Date"

def format_file_size(size_bytes):
    """Formats a file size in bytes into a human-readable string (KB, MB, GB)."""
    if not isinstance(size_bytes, (int, float)) or size_bytes < 0:
        return "N/A"
    if size_bytes == 0:
        return "0 B"
    power = 1024
    n = 0
    power_labels = {0: '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size_bytes >= power and n < len(power_labels):
        size_bytes /= power
        n += 1
    return f"{size_bytes:.2f} {power_labels[n]}"
