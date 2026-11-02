import torch


print("=" * 70)
print("PYTORCH / GPU CHECK")
print("=" * 70)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"GPU count       : {torch.cuda.device_count()}")

    properties = torch.cuda.get_device_properties(0)

    print(
        f"GPU memory      : "
        f"{properties.total_memory / (1024 ** 3):.2f} GB"
    )

    print()
    print("✅ GPU IS AVAILABLE")
    print("✅ Ready for GPU training")

else:
    print()
    print("❌ CUDA GPU NOT AVAILABLE")
    print("Training would run on CPU.")

print("=" * 70)