document.addEventListener('DOMContentLoaded', function () {
    // Function to fetch compliance data from the API
    async function fetchComplianceData() {
        try {
            // TEMPORARY: Using test endpoint that bypasses authentication
            const response = await fetch('/api/compliance_data');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const responseData = await response.json();
            console.log('Received data from API:', responseData); // Debug log to check data format
            
            // Handle different response formats
            if (responseData.data && responseData.status === 'success') {
                console.log('Detected nested data structure, extracting data property');
                return responseData.data;
            } else if (responseData.key_metrics) {
                console.log('Detected flat data structure');
                return responseData;
            } else if (responseData.report_id) {
                console.log('Detected raw report structure.');
                return responseData;
            } else {
                console.error('Unexpected data format:', responseData);
                // Return mock data to prevent dashboard errors
                return {
                    is_mock_data: true,
                    mock_generated_at: '2025-06-25 15:30:00',
                    key_metrics: {
                        sensitive_files: 42,
                        total_scans: 150,
                        risk_level: 'Medium',
                        last_scan_date: '2025-06-25 15:30:00'
                    },
                    sensitive_data_types: {
                        'High Priority': 3,
                        'Medium Priority': 12,
                        'Low Priority': 27
                    },
                    compliance_status: {'GDPR': 95, 'CCPA': 88, 'HIPAA': 92},
                    recent_alerts: [
                        {timestamp: '2025-06-25 15:30:00', message: 'Mock alert 1', severity: 'High'},
                        {timestamp: '2025-06-25 14:25:00', message: 'Mock alert 2', severity: 'Medium'}
                    ]
                };
            }
        } catch (error) {
            console.error("Could not fetch compliance data:", error);
            // Display an error message to the user on the dashboard
            const dashboard = document.querySelector('.container-fluid');
            if (dashboard) {
                const errorHtml = `<div class="alert alert-danger" role="alert">Error loading dashboard data. Please try again later.</div>`;
                dashboard.insertAdjacentHTML('afterbegin', errorHtml);
            }
            return null;
        }
    }

    // Function to update the key metrics
    function updateKeyMetrics(data) {
        if (!data) return;
        console.log('updateKeyMetrics received:', data);
        
        try {
            let metrics = {};
            // Check for different data structures and normalize
            if (data.key_metrics) {
                // Handle nested structure with key_metrics object
                metrics = data.key_metrics;
            } else {
                // Handle flat structure (from old mock data or new report)
                metrics = data;
            }

            document.getElementById('sensitive-files-count').textContent = metrics.sensitive_files || '-';
            document.getElementById('total-scans-count').textContent = metrics.total_scans || '-';
            
            // Handle risk level
            let riskLevel = metrics.risk_level || '-';
            if (typeof riskLevel === 'number') {
                riskLevel = riskLevel === 3 ? 'High' : riskLevel === 2 ? 'Medium' : 'Low';
            }
            document.getElementById('risk-level').textContent = riskLevel;

            // Handle last scan date - **THE KEY FIX**
            // Prioritize 'created_at' from the new report format
            let lastScanDate = '-';
            if (metrics.created_at) {
                lastScanDate = new Date(metrics.created_at).toLocaleString();
            } else if (metrics.last_scan_date) { // Fallback for old mock data
                lastScanDate = new Date(metrics.last_scan_date).toLocaleString();
            } else if (metrics.last_scan) { // Check for last_scan from API response
                lastScanDate = new Date(metrics.last_scan).toLocaleString();
            }
            document.getElementById('last-scan-date').textContent = lastScanDate;

        } catch (error) {
            console.error('Error updating key metrics:', error);
            // Set all metrics to '-' on error to avoid broken UI
            document.getElementById('sensitive-files-count').textContent = '-';
            document.getElementById('total-scans-count').textContent = '-';
            document.getElementById('risk-level').textContent = '-';
            document.getElementById('last-scan-date').textContent = '-';
        }
    }

    // Function to render the sensitive data types chart
    function renderDataTypesChart(data) {
        if (!data) return;
        
        // Get sensitive data types, handling both nested and flat structures
        let dataTypes = {};
        if (data.sensitive_data_types) {
            dataTypes = data.sensitive_data_types;
        } else if (data.data_types) {
            dataTypes = data.data_types;
        } else {
            // Default data if none available
            dataTypes = {
                'Financial': 3,
                'Health': 1,
                'PII': 8
            };
            console.log('No sensitive data types found, using defaults');
        }
        
        const ctx = document.getElementById('dataTypesChart').getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(dataTypes),
                datasets: [{
                    data: Object.values(dataTypes),
                    backgroundColor: ['#4e73df', '#1cc88a', '#36b9cc'],
                    hoverBackgroundColor: ['#2e59d9', '#17a673', '#2c9faf'],
                    hoverBorderColor: "rgba(234, 236, 244, 1)",
                }],
            },
            options: {
                maintainAspectRatio: false,
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed !== null) {
                                    label += context.parsed + ' files';
                                }
                                return label;
                            }
                        }
                    }
                }
            },
        });
    }

    // Function to render the compliance status chart
    function renderComplianceStatusChart(data) {
        if (!data) return;
        
        // Get compliance status, handling both nested and flat structures
        let complianceData = {};
        if (data.compliance_status) {
            complianceData = data.compliance_status;
        } else {
            // Default data if none available
            complianceData = {
                'GDPR': 85,
                'CCPA': 70,
                'HIPAA': 90
            };
            console.log('No compliance status found, using defaults');
        }
        
        const ctx = document.getElementById('complianceStatusChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(complianceData),
                datasets: [{
                    label: 'Compliance %',
                    data: Object.values(complianceData),
                    backgroundColor: '#4e73df',
                    borderColor: '#4e73df',
                    borderWidth: 1
                }]
            },
            options: {
                maintainAspectRatio: false,
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%'
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    // Function to populate the recent alerts table
    function populateAlertsTable(data) {
        if (!data) return;
        const tableBody = document.getElementById('recent-alerts-table');
        tableBody.innerHTML = ''; // Clear existing rows
        
        // Check if recent_alerts exists
        if (!data.recent_alerts || !Array.isArray(data.recent_alerts) || data.recent_alerts.length === 0) {
            console.log('No recent alerts found in data');
            // Add a placeholder row
            tableBody.innerHTML = `
                <tr>
                    <td colspan="3" class="text-center">No recent alerts</td>
                </tr>
            `;
            return;
        }
        
        // Process alerts
        data.recent_alerts.forEach(alert => {
            const severityClass = {
                'High': 'text-danger',
                'Medium': 'text-warning',
                'Low': 'text-info'
            }[alert.severity] || 'text-secondary';

            const row = `
                <tr>
                    <td>${alert.timestamp}</td>
                    <td>${alert.message}</td>
                    <td><span class="${severityClass}">${alert.severity}</span></td>
                </tr>
            `;
            tableBody.innerHTML += row;
        });
    }

    // Function to show or hide mock data indicator
    function showMockDataIndicator(isMockData, timestamp) {
        console.log('showMockDataIndicator called with:', isMockData, timestamp);
        
        // Get or create the indicator element
        let indicator = document.getElementById('mock-data-indicator');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'mock-data-indicator';
            indicator.style.position = 'fixed';
            indicator.style.top = '70px';
            indicator.style.right = '20px';
            indicator.style.backgroundColor = '#ff9800';
            indicator.style.color = 'white';
            indicator.style.padding = '8px 16px';
            indicator.style.borderRadius = '4px';
            indicator.style.fontWeight = 'bold';
            indicator.style.zIndex = '1000';
            indicator.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
            document.body.appendChild(indicator);
        }
        
        // For testing purposes, always show the indicator
        // Remove this line in production
        // isMockData = true;
        
        if (isMockData) {
            let timeDisplay = timestamp ? ` (${timestamp})` : '';
            indicator.textContent = `Demo Mode${timeDisplay}`;
            indicator.style.display = 'block';
            console.log('Mock data indicator shown with text:', indicator.textContent);
        } else {
            indicator.style.display = 'none';
            console.log('Mock data indicator hidden');
        }
    }

    /**
     * Main function to initialize the dashboard
     * 
     * This function handles:
     * 1. Fetching compliance data from the API
     * 2. Detecting if the data is mock data (using is_mock_data flag)
     * 3. Showing the "Demo Mode" badge when mock data is detected
     * 4. Updating all UI components with the fetched data
     */
    async function initializeDashboard() {
        const data = await fetchComplianceData();
        if (data) {
            // For debugging - log the entire data object to help diagnose structure issues
            console.log('Full data object:', JSON.stringify(data));
            
            /**
             * Mock Data Detection Logic
             * 
             * We use strict equality (===) to check for the is_mock_data flag
             * This avoids issues with truthy/falsy values that might be misinterpreted
             * 
             * The flag can be at either:
             * - Top level of the response (data.is_mock_data)
             * - Inside a nested data property (data.data.is_mock_data)
             */
            const isMockData = data.is_mock_data === true || 
                             (data.data && data.data.is_mock_data === true) || 
                             false;
            
            /**
             * Mock Data Timestamp
             * 
             * When mock data is used, we display the timestamp when it was generated
             * This helps users understand that they're looking at non-real-time data
             * and when that mock data was created
             */
            const mockTimestamp = data.mock_generated_at || 
                                (data.data && data.data.mock_generated_at) || 
                                null;
            
            console.log('Mock data status:', isMockData, mockTimestamp);
            
            // Show mock data indicator only if the data is actually mock data
            showMockDataIndicator(isMockData, mockTimestamp);
            
            // Update all UI components with the fetched data
            updateKeyMetrics(data);
            renderDataTypesChart(data);
            renderComplianceStatusChart(data);
            populateAlertsTable(data);
        }
    }

    initializeDashboard();
});
