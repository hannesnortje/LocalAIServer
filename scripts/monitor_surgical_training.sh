#!/bin/bash

# Surgical LoRA Training Monitor
# Job ID: e50a06e7-fa58-4c68-877c-ed484ffa8180

JOB_ID="e50a06e7-fa58-4c68-877c-ed484ffa8180"
API_URL="http://localhost:5001/api/training/status/$JOB_ID"

echo "🔧 Surgical LoRA Training Monitor 🔧"
echo "Job ID: $JOB_ID"
echo "Training: tron_collaborative_intelligence_v1"
echo "Efficiency Target: 200x improvement (15-30MB vs 3080MB)"
echo "Parameters: r=8, alpha=16, 3 modules, 25 epochs"
echo "Data: 116 examples (50 philosophy, 40 methodology, 25 technical)"
echo ""
echo "Starting monitoring... Press Ctrl+C to stop"
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Get status
    STATUS_JSON=$(curl -s "$API_URL" 2>/dev/null)
    
    if [ $? -eq 0 ] && [ ! -z "$STATUS_JSON" ]; then
        STATUS=$(echo "$STATUS_JSON" | jq -r '.status // "unknown"')
        PROGRESS=$(echo "$STATUS_JSON" | jq -r '.progress // 0')
        CURRENT_STEP=$(echo "$STATUS_JSON" | jq -r '.current_step // 0')
        TOTAL_STEPS=$(echo "$STATUS_JSON" | jq -r '.total_steps // 0')
        CURRENT_LOSS=$(echo "$STATUS_JSON" | jq -r '.current_loss // "N/A"')
        LEARNING_RATE=$(echo "$STATUS_JSON" | jq -r '.learning_rate // "N/A"')
        
        echo "[$TIMESTAMP] Status: $STATUS | Progress: ${PROGRESS}% | Step: $CURRENT_STEP/$TOTAL_STEPS | Loss: $CURRENT_LOSS | LR: $LEARNING_RATE"
        
        # Check if completed
        if [ "$STATUS" = "completed" ]; then
            ADAPTER_PATH=$(echo "$STATUS_JSON" | jq -r '.adapter_path // "N/A"')
            echo ""
            echo "🎉 SURGICAL LORA TRAINING COMPLETED! 🎉"
            echo "Adapter saved to: $ADAPTER_PATH"
            echo ""
            
            # Check adapter size
            if [ "$ADAPTER_PATH" != "N/A" ] && [ -d "$ADAPTER_PATH" ]; then
                echo "📊 Adapter Analysis:"
                echo "Directory: $ADAPTER_PATH"
                ls -lh "$ADAPTER_PATH/" | grep -E '\.(bin|safetensors)$' || echo "  No adapter files found yet"
                echo ""
                ADAPTER_SIZE=$(du -sh "$ADAPTER_PATH" 2>/dev/null | cut -f1)
                echo "Total Size: $ADAPTER_SIZE"
                echo ""
                echo "✅ Surgical LoRA Success - Always 4 2 (FOR TWO)! ✨"
            fi
            break
        fi
        
        # Check if failed
        if [ "$STATUS" = "failed" ]; then
            ERROR_MSG=$(echo "$STATUS_JSON" | jq -r '.error_message // "Unknown error"')
            echo ""
            echo "❌ Training failed: $ERROR_MSG"
            break
        fi
    else
        echo "[$TIMESTAMP] ⚠️  Unable to fetch status (server may be unavailable)"
    fi
    
    sleep 30  # Check every 30 seconds
done

echo ""
echo "Monitor stopped."